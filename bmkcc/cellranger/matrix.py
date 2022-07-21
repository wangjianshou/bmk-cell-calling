#!/usr/bin/env python
#
# Copyright (c) 2018 10X Genomics, Inc. All rights reserved.
#
from __future__ import absolute_import, annotations, division, print_function
from collections import OrderedDict
import copy
import pathlib
import os.path
import h5py as h5
import numpy as np
import scipy.io as sp_io
import pandas as pd

pd.set_option("compute.use_numexpr", False)
import scipy.sparse as sp_sparse
from six import ensure_binary, ensure_str, ensure_text

from bmkcc.cellranger.wrapped_tables import tables
import bmkcc.tenkit.safe_json as tk_safe_json
import bmkcc.cellranger.h5_constants as h5_constants
import bmkcc.cellranger.rna.library as rna_library
from bmkcc.cellranger.feature_ref import FeatureReference, FeatureDef, GENOME_FEATURE_TAG
import bmkcc.cellranger.utils as cr_utils
import bmkcc.cellranger.hdf5 as cr_h5
import bmkcc.cellranger.io as cr_io
import bmkcc.cellranger.sparse as cr_sparse
import bmkcc.cellranger.bisect as cr_bisect  # pylint: disable=no-name-in-module

from typing import (
    Any,
    Collection,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

HDF5_COMPRESSION = "gzip"
# Number of elements per chunk. Here, 1 MiB / (12 bytes)
HDF5_CHUNK_SIZE = 80000

DEFAULT_DATA_DTYPE = "int32"


# some helper functions from stats
def sum_sparse_matrix(matrix, axis: int = 0) -> np.ndarray:
    """Sum a sparse matrix along an axis."""
    axis_sum = np.asarray(matrix.sum(axis=axis))  # sum along given axis
    max_dim = np.prod(axis_sum.shape)  # get the max dimension
    return axis_sum.reshape((max_dim,))  # reshape accordingly


def top_n(array: np.ndarray, n: int) -> Tuple[int, Any]:
    """Retrieve the N largest elements and their positions in a numpy ndarray.

    Args:
       array (numpy.ndarray): Array
       n (int): Number of elements

    Returns:
       list of tuple of (int, x): Tuples are (original index, value).
    """
    indices = np.argpartition(array, -n)[-n:]
    indices = indices[np.argsort(array[indices])]
    return zip(indices, array[indices])


MATRIX_H5_FILETYPE = "matrix"
MATRIX = "matrix"
MATRIX_H5_VERSION_KEY = "version"
MATRIX_H5_VERSION = 2
SOFTWARE_H5_VERSION_KEY = "software_version"

# used to distinguish from user-defined attrs introduced in aggr
MATRIX_H5_BUILTIN_ATTRS = [
    h5_constants.H5_FILETYPE_KEY,
    MATRIX_H5_VERSION_KEY,
] + h5_constants.H5_METADATA_ATTRS


class NullAxisMatrixError(Exception):
    pass


class CountMatrixView(object):
    """Supports summing a sliced CountMatrix w/o copying the whole thing"""

    def __init__(
        self,
        matrix: CountMatrix,
        feature_indices: Optional[Collection[int]] = None,
        bc_indices: Optional[Collection[int]] = None,
    ):
        self.feature_mask = np.ones(matrix.features_dim, dtype="bool")
        self.bc_mask = np.ones(matrix.bcs_dim, dtype="bool")
        self.matrix = matrix

        if feature_indices is not None:
            self.feature_mask.fill(False)
            self.feature_mask[np.asarray(feature_indices)] = True
        if bc_indices is not None:
            self.bc_mask.fill(False)
            self.bc_mask[np.asarray(bc_indices)] = True

        self._update_feature_ref()

    @property
    def bcs_dim(self) -> int:
        return np.count_nonzero(self.bc_mask)

    @property
    def features_dim(self) -> int:
        return np.count_nonzero(self.feature_mask)

    def _copy(self) -> CountMatrixView:
        """Return a copy of this view"""
        view = CountMatrixView(self.matrix)
        view.bc_mask = np.copy(self.bc_mask)
        view.feature_mask = np.copy(self.feature_mask)
        view._update_feature_ref()
        return view

    def view(self) -> CountMatrixView:
        """Return a copy of this view"""
        return self._copy()

    def sum(self, axis: Optional[int] = None):
        """Sum across an axis."""
        return cr_sparse.sum_masked(self.matrix.m, self.feature_mask, self.bc_mask, axis=axis)

    def count_ge(self, axis: Optional[int], threshold) -> Union[int, np.ndarray]:
        """Count number of elements >= X over an axis"""
        return cr_sparse.count_ge_masked(
            self.matrix.m, self.feature_mask, self.bc_mask, threshold, axis
        )

    def select_barcodes(self, indices: Union[List[int], np.ndarray]):
        """Select a subset of barcodes (by index in the original matrix) and return the resulting view"""
        view = self._copy()
        mask = np.bincount(indices, minlength=len(view.bc_mask)).astype("bool")
        mask[mask > 1] = 1
        view.bc_mask &= mask
        return view

    def select_barcodes_by_seq(self, barcode_seqs):
        # type: (List[bytes]) -> CountMatrixView
        indices = self.matrix.bcs_to_ints(barcode_seqs)
        return self.select_barcodes(indices)

    def select_barcodes_by_gem_group(self, gem_group: int) -> CountMatrixView:
        return self.select_barcodes_by_seq(
            [bc for bc in self.matrix.bcs if gem_group == cr_utils.split_barcode_seq(bc)[1]]
        )

    def _update_feature_ref(self) -> FeatureReference:
        """Make the feature reference consistent with the feature mask"""
        indices = np.flatnonzero(self.feature_mask)
        self.feature_ref = FeatureReference(
            feature_defs=[self.matrix.feature_ref.feature_defs[i] for i in indices],
            all_tag_keys=self.matrix.feature_ref.all_tag_keys,
            target_features=self.matrix.feature_ref.target_features,
        )

    def select_features(self, indices: Union[List[int], np.ndarray]) -> CountMatrixView:
        """Select a subset of features and return the resulting view."""
        view = self._copy()
        mask = np.bincount(indices, minlength=len(view.feature_mask)).astype("bool")
        mask[mask > 1] = 1
        view.feature_mask &= mask
        view._update_feature_ref()
        return view

    def select_features_by_genome(self, genome):
        """Select the subset of gene-expression features for genes in a specific genome"""
        indices = []
        for feature in self.matrix.feature_ref.feature_defs:
            if feature.feature_type == rna_library.DEFAULT_LIBRARY_TYPE:
                if feature.tags["genome"] == genome:
                    indices.append(feature.index)
        return self.select_features(indices)

    def select_features_by_genome_and_types(self, genome, feature_types):
        """Subset the features by types and genome.

        Select the subset of gene-expression features for genes in a specific genome and matching
        one of the types listed in feature_types.
        """
        indices = []
        for feature in self.matrix.feature_ref.feature_defs:
            if feature.feature_type in feature_types:
                include_feature = True
                # Genome test only applies to GEX features
                if feature.feature_type == rna_library.DEFAULT_LIBRARY_TYPE:
                    if feature.tags["genome"] != genome:
                        include_feature = False
                if include_feature:
                    indices.append(feature.index)
        return self.select_features(indices)

    def select_features_by_types(self, feature_types):
        """Subset the features by type.

        Select the subset of gene-expression features for genes in a specific genome and matching
        one of the types listed in feature_types.
        """
        indices = []
        for feature in self.matrix.feature_ref.feature_defs:
            if feature.feature_type in feature_types:
                indices.append(feature.index)
        return self.select_features(indices)

    def select_features_by_type(self, feature_type):
        """Select the subset of features with a particular feature type (e.g. "Gene Expression")"""
        return self.select_features(self.get_feature_indices_by_type(feature_type))

    def select_features_by_ids(self, feature_ids: Iterable[bytes]) -> CountMatrixView:
        return self.select_features(self.matrix.feature_ids_to_ints(feature_ids))

    def get_feature_indices_by_type(self, feature_type: str) -> List[int]:
        """Return the list of indices of features corresponding a feature type"""
        return self.matrix.feature_ref.get_indices_for_type(feature_type)

    def get_genomes(self) -> List[str]:
        """Get a list of the distinct genomes represented by gene expression features"""
        return CountMatrix._get_genomes_from_feature_ref(self.feature_ref)

    def bcs_to_ints(self, bcs: Union[List[bytes], Set[bytes]]) -> List[int]:
        # Only works when we haven't masked barcodes.
        if np.count_nonzero(self.bc_mask) != self.matrix.bcs_dim:
            raise NotImplementedError(
                "Calling bcs_to_ints on a barcode-sliced matrix view is unimplemented"
            )
        return self.matrix.bcs_to_ints(bcs)

    def ints_to_bcs(self, bc_ints: Union[List[int], np.ndarray]) -> List[bytes]:
        if bc_ints is None or len(bc_ints) == 0:
            return []
        sliced_bc_ints = np.flatnonzero(self.bc_mask)
        orig_bc_ints = sliced_bc_ints[np.asarray(bc_ints)]
        return [self.matrix.bcs[i] for i in orig_bc_ints]

    def int_to_feature_id(self, i: int) -> bytes:
        return self.feature_ref.feature_defs[i].id

    def get_shape(self) -> Tuple[int, int]:
        """Return the shape of the sliced matrix"""
        return (np.count_nonzero(self.feature_mask), np.count_nonzero(self.bc_mask))

    def get_num_nonzero(self):
        """Return the number of nonzero entries in the sliced matrix"""
        return self.count_ge(axis=None, threshold=1)

    def get_counts_per_bc(self):
        return self.sum(axis=0)


class CountMatrix(object):
    # pylint: disable=too-many-public-methods
    def __init__(
        self,
        feature_ref: FeatureReference,
        bcs: Union[Collection[bytes], Collection[str]],
        matrix: sp_sparse.spmatrix,
    ):
        # Features (genes, CRISPR gRNAs, antibody barcodes, etc.)
        self.feature_ref = feature_ref
        self.features_dim = len(feature_ref.feature_defs)
        self.feature_ids_map = {
            f.id: f.index for f in feature_ref.feature_defs
        }  # type: Dict[bytes, int]

        # Cell barcodes
        bc_array = np.array(list(bcs), dtype="S", copy=False)
        del bcs
        bc_array.flags.writeable = False
        self.bcs = bc_array  # type: np.ndarray
        (self.bcs_dim,) = self.bcs.shape
        bcs_idx = np.argsort(self.bcs).astype(np.int32)
        bcs_idx.flags.writeable = False
        self.bcs_idx = bcs_idx  # type: np.ndarray

        self.m = matrix  # type: sp_sparse.spmatrix
        assert self.m.shape[1] == len(self.bcs), "Barcodes must be equal to cols of matrix"

    def get_shape(self):
        """Return the shape of the sliced matrix"""
        return self.m.shape

    def get_num_nonzero(self):
        """Return the number of nonzero entries in the sliced matrix"""
        return self.m.nnz

    def view(self):
        """Return a view on this matrix"""
        return CountMatrixView(self)

    @classmethod
    def empty(cls, feature_ref: FeatureReference, bcs: Collection[bytes], dtype=DEFAULT_DATA_DTYPE):
        """Create an empty matrix."""
        matrix = sp_sparse.lil_matrix((len(feature_ref.feature_defs), len(bcs)), dtype=dtype)
        return cls(feature_ref=feature_ref, bcs=bcs, matrix=matrix)

    @staticmethod
    def from_legacy_v1_h5(h5_file: h5.File) -> CountMatrix:
        """Create a CountMatrix from a legacy h5py.File (format version 1)"""

        genome_arrays = []
        gene_id_arrays = []
        gene_name_arrays = []
        bc_idx_arrays = []
        feat_idx_arrays = []
        data_arrays = []

        # Map barcode string to column index in new matrix
        barcode_map = OrderedDict()

        # Construct a genome-concatenated matrix and FeatureReference
        for genome_idx, genome in enumerate(h5_file.keys()):
            g = h5_file[genome]

            n_genes = sum(len(x) for x in gene_id_arrays)

            # Offset the row (gene) indices by the number of genes seen so far
            feat_idx_arrays.append(g["indices"][:] + n_genes)

            # Offset the col (barcode) indices by the number of nonzero elements seen so far

            # Map barcode (column) indices to a single unique barcode space
            barcodes = g["barcodes"][:]
            for bc in barcodes:
                if bc not in barcode_map:
                    barcode_map[bc] = len(barcode_map)

            remapped_col_inds = np.fromiter(
                (barcode_map[bc] for bc in barcodes), count=len(barcodes), dtype="uint64"
            )

            indptr = g["indptr"][:]
            assert len(indptr) == 1 + len(remapped_col_inds)

            if genome_idx == 0:
                # For the first set of barcodes encountered, there should
                # be no change in their new indices.
                assert np.array_equal(remapped_col_inds, np.arange(len(indptr) - 1))

            # Convert from CSC to COO by expanding the indptr array out

            nz_elems_per_bc = np.diff(indptr)
            assert len(nz_elems_per_bc) == len(g["barcodes"])

            bc_idx = np.repeat(remapped_col_inds, nz_elems_per_bc)
            assert len(bc_idx) == len(g["indices"])
            assert len(bc_idx) == len(g["data"])

            bc_idx_arrays.append(bc_idx)
            data_arrays.append(g["data"][:])

            gene_id_arrays.append(g["genes"][:])
            gene_name_arrays.append(g["gene_names"][:])
            genome_arrays.append(np.repeat(genome, len(g["genes"])))

        genomes = np.concatenate(genome_arrays)
        gene_ids = np.concatenate(gene_id_arrays)
        gene_names = np.concatenate(gene_name_arrays)

        # Construct FeatureReference
        feature_defs = []
        for (gene_id, gene_name, genome) in zip(gene_ids, gene_names, genomes):
            feature_defs.append(
                FeatureDef(
                    index=len(feature_defs),
                    id=gene_id,
                    name=gene_name,
                    feature_type=rna_library.GENE_EXPRESSION_LIBRARY_TYPE,
                    tags={GENOME_FEATURE_TAG: genome},
                )
            )
        feature_ref = FeatureReference(feature_defs, [GENOME_FEATURE_TAG])

        i = np.concatenate(feat_idx_arrays)
        j = np.concatenate(bc_idx_arrays)
        data = np.concatenate(data_arrays)

        assert isinstance(barcode_map, OrderedDict)

        matrix = sp_sparse.csc_matrix((data, (i, j)), shape=(len(gene_ids), len(barcode_map)))

        return CountMatrix(
            feature_ref, barcode_map.keys(), matrix  # pylint: disable=dict-keys-not-iterating
        )

    @staticmethod
    def from_legacy_mtx(genome_dir):
        barcodes_tsv = ensure_binary(os.path.join(genome_dir, "barcodes.tsv"))
        genes_tsv = ensure_binary(os.path.join(genome_dir, "genes.tsv"))
        matrix_mtx = ensure_binary(os.path.join(genome_dir, "matrix.mtx"))
        for filepath in [barcodes_tsv, genes_tsv, matrix_mtx]:
            if not os.path.exists(filepath):
                raise IOError("Required file not found: %s" % filepath)
        barcodes = pd.read_csv(
            barcodes_tsv.encode(),
            delimiter="\t",
            header=None,
            usecols=[0],
            dtype=bytes,
            converters={0: ensure_binary},
        ).values.squeeze()
        genes = pd.read_csv(
            genes_tsv.encode(),
            delimiter="\t",
            header=None,
            usecols=[0],
            dtype=bytes,
            converters={0: ensure_binary},
        ).values.squeeze()
        feature_defs = [
            FeatureDef(idx, gene_id, None, "Gene Expression", {})
            for (idx, gene_id) in enumerate(genes)
        ]
        feature_ref = FeatureReference(feature_defs, [])

        matrix = sp_io.mmread(matrix_mtx)
        mat = CountMatrix(feature_ref, barcodes, matrix)
        return mat

    @staticmethod
    def from_v3_mtx(genome_dir):
        barcodes_tsv = ensure_str(os.path.join(genome_dir, "barcodes.tsv.gz"))
        features_tsv = ensure_str(os.path.join(genome_dir, "features.tsv.gz"))
        matrix_mtx = ensure_str(os.path.join(genome_dir, "matrix.mtx.gz"))
        for filepath in [barcodes_tsv, features_tsv, matrix_mtx]:
            if not os.path.exists(filepath):
                raise IOError("Required file not found: {}".format(filepath))
        barcodes = pd.read_csv(
            barcodes_tsv, delimiter="\t", header=None, usecols=[0], dtype=bytes
        ).values.squeeze()
        features = pd.read_csv(features_tsv, delimiter="\t", header=None)

        feature_defs = []
        for (idx, (_, r)) in enumerate(features.iterrows()):
            fd = FeatureDef(idx, r[0], r[1], r[2], {})
            feature_defs.append(fd)

        feature_ref = FeatureReference(feature_defs, [])

        matrix = sp_io.mmread(matrix_mtx)
        mat = CountMatrix(feature_ref, barcodes, matrix)
        return mat

    @staticmethod
    def load_mtx(mtx_dir):
        legacy_fn = os.path.join(mtx_dir, "genes.tsv")
        v3_fn = os.path.join(mtx_dir, "features.tsv.gz")
        if os.path.exists(legacy_fn):
            return CountMatrix.from_legacy_mtx(mtx_dir)

        if os.path.exists(v3_fn):
            return CountMatrix.from_v3_mtx(mtx_dir)

        raise IOError("Not a valid path to a feature-barcode mtx directory: '%s'" % str(mtx_dir))

    def feature_id_to_int(self, feature_id):
        # type: (bytes) -> int
        if not isinstance(feature_id, bytes):
            raise KeyError(
                "feature_id {} must be bytes, but was {}".format(feature_id, type(feature_id))
            )
        if feature_id not in self.feature_ids_map:
            raise KeyError(
                "Specified feature ID not found in matrix: {}".format(ensure_str(feature_id))
            )
        return self.feature_ids_map[feature_id]

    def feature_ids_to_ints(self, feature_ids: Iterable[bytes]) -> List[int]:
        return sorted(self.feature_id_to_int(fid) for fid in feature_ids)

    def feature_id_to_name(self, feature_id: bytes) -> str:
        idx = self.feature_id_to_int(feature_id)
        return self.feature_ref.feature_defs[idx].name

    def int_to_feature_id(self, i: int) -> bytes:
        return self.feature_ref.feature_defs[i].id

    def int_to_feature_name(self, i: int) -> str:
        return self.feature_ref.feature_defs[i].name

    def bc_to_int(self, bc: bytes) -> int:
        """Get the integer index for a barcode.

        Args:
            bc (bytes): The barcode to search for.

        Raises:
            ValueError: `barcode` was not bytes.
            KeyError: `barcode` was not found in the set.

        Returns:
            int: the barcode index.
        """
        # Don't do a conversion for bc, because it's much more efficient to just
        # load the barcodes in the correct format in the first place.  Just
        # raise an exception so we get a nice stack trace and clear error.
        if not isinstance(bc, bytes):
            raise ValueError("barcode must be bytes, but was {}".format(type(bc)))
        j = cr_bisect.bisect_left(self.bcs_idx, bc, self.bcs)
        if j >= self.bcs_dim or self.bcs[j] != bc:
            raise KeyError("Specified barcode not found in matrix: %s" % ensure_str(bc))
        return j

    def bcs_to_ints(
        self, bcs: Union[List[bytes], Set[bytes]], return_sorted: bool = True
    ) -> List[int]:
        if isinstance(bcs, set):
            bcs = list(bcs)
        n = len(self.bcs_idx) - 1
        as_str_array = np.array(bcs, dtype=bytes)
        bcs_array = np.argsort(as_str_array)
        indices = [-1] * len(bcs)
        guess = 0
        for i, v in enumerate(bcs_array):
            bc = as_str_array[v]
            if self.bcs[guess] == bc:
                pos = guess
            else:
                pos = self.bc_to_int(bc)
                guess = min(pos + 1, n)
            if return_sorted:
                indices[i] = pos
            else:
                indices[v] = pos
        return indices

    def int_to_bc(self, j: int) -> bytes:
        return self.bcs[j]

    def ints_to_bcs(self, jj: Iterable[int]) -> List[bytes]:
        return [self.int_to_bc(j) for j in jj]

    def add(self, feature_id: bytes, bc: bytes, value=1) -> None:
        """Add a count."""
        i, j = self.feature_id_to_int(feature_id), self.bc_to_int(bc)
        self.m[i, j] += value

    def get(self, feature_id: bytes, bc: bytes) -> None:
        i, j = self.feature_id_to_int(feature_id), self.bc_to_int(bc)
        return self.m[i, j]

    def merge(self, other: CountMatrix) -> None:
        """Merge this matrix with another CountMatrix"""
        assert self.features_dim == other.features_dim
        self.m += other.m

    def sort_indices(self) -> None:
        self.tocsc()
        if not self.m.has_sorted_indices:
            self.m.sort_indices()

    def save_dense_csv(self, filename):
        """Save this matrix to a dense CSV file."""
        dense_cm = pd.DataFrame(
            self.m.toarray(),
            index=[ensure_str(f.id) for f in self.feature_ref.feature_defs],
            columns=[ensure_str(bc) for bc in self.bcs],
        )
        dense_cm.to_csv(filename, index=True, header=True)

    def save_h5_file(self, filename, extra_attrs={}, sw_version=None):
        """Save this matrix to an HDF5 file, optionally with SW version"""
        with h5.File(ensure_binary(filename), "w") as f:
            f.attrs[h5_constants.H5_FILETYPE_KEY] = MATRIX_H5_FILETYPE
            f.attrs[MATRIX_H5_VERSION_KEY] = MATRIX_H5_VERSION
            # set the software version key only if it is supplied
            if sw_version:
                f.attrs[SOFTWARE_H5_VERSION_KEY] = sw_version

            # Set optional top-level attributes
            for (k, v) in extra_attrs.items():
                cr_h5.set_hdf5_attr(f, k, v)

            group = f.create_group(MATRIX)
            self.save_h5_group(group)

    def save_h5_group(self, group: h5.Group) -> None:
        """Save this matrix to an HDF5 (h5py) group."""
        self.sort_indices()

        # Save the feature reference
        feature_ref_group = group.create_group(h5_constants.H5_FEATURE_REF_ATTR)
        self.feature_ref.to_hdf5(feature_ref_group)

        # Store barcode sequences as array of ASCII strings
        cr_h5.create_hdf5_string_dataset(
            group, h5_constants.H5_BCS_ATTR, self.bcs, compression=True
        )

        for attr, dtype in h5_constants.H5_MATRIX_ATTRS.items():
            arr = np.array(getattr(self.m, attr), dtype=dtype)
            group.create_dataset(
                attr,
                data=arr,
                chunks=(HDF5_CHUNK_SIZE,),
                maxshape=(None,),
                compression=HDF5_COMPRESSION,
                shuffle=True,
            )

    @staticmethod
    def load_dims(group: h5.Group) -> Tuple[int, int, int]:
        """Load the matrix shape from an HDF5 group"""
        (rows, cols) = group[h5_constants.H5_MATRIX_SHAPE_ATTR][:]
        entries = len(group[h5_constants.H5_MATRIX_DATA_ATTR])
        return (rows, cols, entries)

    @staticmethod
    def load_dims_from_h5(filename) -> Tuple[int, int, int]:
        """Load the matrix shape from an HDF5 file"""
        filename = ensure_binary(filename)
        h5_version = CountMatrix.get_format_version_from_h5(filename)
        if h5_version == 1:
            return CountMatrix._load_dims_from_legacy_v1_h5(filename)
        else:
            with h5.File(filename, "r") as f:
                return CountMatrix.load_dims(f[MATRIX])

    @staticmethod
    def _load_dims_from_legacy_v1_h5_handle(f: h5.File) -> Tuple[int, int, int]:
        # legacy format
        genomes = f.keys()
        num_nonzero_entries = 0
        num_gene_ids = 0
        barcodes = set()
        for genome in genomes:
            g = f[genome]
            num_nonzero_entries += len(g["data"])
            num_gene_ids += len(g["genes"])
            barcodes.update(g["barcodes"])
        return (num_gene_ids, len(barcodes), num_nonzero_entries)

    @staticmethod
    def _load_dims_from_legacy_v1_h5(filename) -> Tuple[int, int, int]:
        """Load the matrix shape from a legacy h5py.File (format version 1)"""
        with h5.File(ensure_binary(filename), "r") as f:
            return CountMatrix._load_dims_from_legacy_v1_h5_handle(f)

    @staticmethod
    def get_mem_gb_from_matrix_dim(num_barcodes: int, nonzero_entries: int) -> float:
        """ Estimate memory usage of loading a matrix. """
        matrix_mem_gb = float(nonzero_entries) / h5_constants.NUM_MATRIX_ENTRIES_PER_MEM_GB
        # We store a list and a dict of the whitelist. Based on empirical obs.
        matrix_mem_gb += float(num_barcodes) / h5_constants.NUM_MATRIX_BARCODES_PER_MEM_GB

        return h5_constants.MATRIX_MEM_GB_MULTIPLIER * np.ceil(matrix_mem_gb)

    @staticmethod
    def get_mem_gb_from_group(group: h5.Group) -> float:
        """Estimate memory usage from an HDF5 group."""
        _, num_bcs, nonzero_entries = CountMatrix.load_dims(group)
        return CountMatrix.get_mem_gb_from_matrix_dim(num_bcs, nonzero_entries)

    @staticmethod
    def get_mem_gb_crconverter_estimate_from_h5(filename) -> float:
        """Estimate the amount of memory needed for the crconverter program to process a GEX matrix.

        See the commit message for a full explanation of this process.
        """
        filename = ensure_binary(filename)
        h5_version = CountMatrix.get_format_version_from_h5(filename)
        if h5_version == 1:
            # Estimate memory usage from a legacy h5py.File (format version 1)
            _, _, nonzero_entries = CountMatrix._load_dims_from_legacy_v1_h5(filename)
        else:
            with h5.File(filename, "r") as f:
                _, _, nonzero_entries = CountMatrix.load_dims(f[MATRIX])
        return CountMatrix._get_mem_gb_crconverter_estimate_from_nnz(nonzero_entries)

    @staticmethod
    def _get_mem_gb_crconverter_estimate_from_nnz(nnz: int) -> float:
        # Empirically determined baseline + ~85 bytes per nnz
        estimate = 1.691 + 8.537e-8 * nnz
        # Arbitrary safety factor
        estimate *= 1.5
        return np.ceil(estimate)

    @staticmethod
    def get_mem_gb_from_matrix_h5(filename) -> float:
        """Estimate memory usage from an HDF5 file."""
        filename = ensure_binary(filename)
        h5_version = CountMatrix.get_format_version_from_h5(filename)
        if h5_version == 1:
            return CountMatrix._get_mem_gb_from_legacy_v1_h5(filename)
        else:
            with h5.File(filename, "r") as f:
                return CountMatrix.get_mem_gb_from_group(f[MATRIX])

    @staticmethod
    def _get_mem_gb_from_legacy_v1_h5(filename) -> float:
        """Estimate memory usage from a legacy h5py.File (format version 1)"""
        _, num_bcs, nonzero_entries = CountMatrix._load_dims_from_legacy_v1_h5(filename)
        return CountMatrix.get_mem_gb_from_matrix_dim(num_bcs, nonzero_entries)

    @classmethod
    def load(cls, group: h5.Group) -> CountMatrix:
        """Load from an HDF5 group."""
        feature_ref = CountMatrix.load_feature_ref_from_h5_group(group)
        bcs = cls.load_bcs_from_h5_group(group)

        shape = group[h5_constants.H5_MATRIX_SHAPE_ATTR][:]
        data = group[h5_constants.H5_MATRIX_DATA_ATTR][:]
        indices = group[h5_constants.H5_MATRIX_INDICES_ATTR][:]
        indptr = group[h5_constants.H5_MATRIX_INDPTR_ATTR][:]

        # Check to make sure indptr increases monotonically (to catch overflow bugs)
        assert np.all(np.diff(indptr) >= 0)

        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)

        return cls(feature_ref=feature_ref, bcs=bcs, matrix=matrix)

    @staticmethod
    def load_bcs_from_h5_group(group: h5.Group) -> List[bytes]:
        """Load just the barcode sequences from an h5 group."""
        if group[h5_constants.H5_BCS_ATTR].shape is not None:
            return list(group[h5_constants.H5_BCS_ATTR][:])
        return []

    @staticmethod
    def load_bcs_from_h5(filename) -> List[bytes]:
        """Load just the barcode sequences from an HDF5 group. """
        filename = ensure_binary(filename)
        h5_version = CountMatrix.get_format_version_from_h5(filename)
        if h5_version == 1:
            return CountMatrix._load_bcs_from_legacy_v1_h5(filename)
        else:
            with h5.File(filename, "r") as f:
                return CountMatrix.load_bcs_from_h5_group(f[MATRIX])

    @staticmethod
    def _load_bcs_from_legacy_v1_h5(filename) -> List[bytes]:
        """Load just the barcode sequences from a legacy h5py.File (format version 1)"""
        with h5.File(ensure_binary(filename), "r") as f:
            genomes = f.keys()
            barcodes = set()  # type: set[bytes]
            for genome in genomes:
                group = f[genome]
                barcodes.update(group["barcodes"])
            return list(barcodes)

    @staticmethod
    def load_bcs_from_h5_file(filename) -> List[bytes]:
        with h5.File(ensure_binary(filename), "r") as f:
            if (
                h5_constants.H5_FILETYPE_KEY not in f.attrs
                or f.attrs[h5_constants.H5_FILETYPE_KEY] != MATRIX_H5_FILETYPE
            ):
                raise ValueError("HDF5 file is not a valid matrix HDF5 file.")

            if MATRIX_H5_VERSION_KEY in f.attrs:
                version = f.attrs[MATRIX_H5_VERSION_KEY]
            else:
                version = 1

            if version > MATRIX_H5_VERSION:
                raise ValueError(
                    "Matrix HDF5 file format version (%d) is a newer version that is not supported by this version of the software."
                    % version
                )
            if version < MATRIX_H5_VERSION:
                raise ValueError(
                    "Matrix HDF5 file format version (%d) is an older version that is no longer supported."
                    % version
                )

            if "matrix" not in f.keys():
                raise ValueError('Could not find the "matrix" group inside the matrix HDF5 file.')

            return CountMatrix.load_bcs_from_h5_group(f["matrix"])

    @staticmethod
    def load_library_types_from_h5_file(filename) -> Set[str]:
        """ Return a set of all library types defined in the Feature Reference """
        with h5.File(ensure_binary(filename), "r") as f:
            version = CountMatrix._get_format_version_from_handle(f)
            if version < MATRIX_H5_VERSION:
                # Only GEX supported, check for any data and return GEX if any exists
                (gene_count, _, _) = CountMatrix._load_dims_from_legacy_v1_h5_handle(f)
                if gene_count > 0:
                    return {rna_library.GENE_EXPRESSION_LIBRARY_TYPE}
                else:
                    return set()
            else:
                feature_ref = CountMatrix.load_feature_ref_from_h5_group(f[MATRIX])
                return set(f.feature_type for f in feature_ref.feature_defs)

    @staticmethod
    def load_feature_ref_from_h5_group(group: h5.Group) -> FeatureReference:
        """Load just the FeatureRef from an h5py.Group."""
        feature_group = group[h5_constants.H5_FEATURE_REF_ATTR]
        return FeatureReference.from_hdf5(feature_group)

    @staticmethod
    def load_feature_ref_from_h5_file(filename) -> FeatureReference:
        """Load just the FeatureRef from a matrix HDF5 file."""
        with h5.File(ensure_binary(filename), "r") as f:
            version = CountMatrix._get_format_version_from_handle(f)
            if version < MATRIX_H5_VERSION:
                raise IOError("Direct Feature Ref reading not supported for older H5 files.")
            else:
                return CountMatrix.load_feature_ref_from_h5_group(f[MATRIX])

    def tolil(self):
        if type(self.m) is not sp_sparse.lil_matrix:
            self.m = self.m.tolil()

    def tocoo(self):
        if type(self.m) is not sp_sparse.coo_matrix:
            self.m = self.m.tocoo()

    def tocsc(self):
        # Convert from lil to csc matrix for efficiency when analyzing data
        if type(self.m) is not sp_sparse.csc_matrix:
            self.m = self.m.tocsc()

    def select_axes_above_threshold(
        self, threshold: int = 0
    ) -> Tuple[CountMatrix, np.ndarray, np.ndarray]:
        """Select axes with sums greater than the threshold value.

        Returns:
            (CountMatrix, np.array of int, np.array of int):
                New count matrix, non-zero bc indices, feat indices
        """

        new_mat = copy.deepcopy(self)

        nonzero_bcs = np.flatnonzero(new_mat.get_counts_per_bc() > threshold)
        if self.bcs_dim > len(nonzero_bcs):
            new_mat = new_mat.select_barcodes(nonzero_bcs)

        nonzero_features = np.flatnonzero(new_mat.get_counts_per_feature() > threshold)
        if new_mat.features_dim > len(nonzero_features):
            new_mat = new_mat.select_features(nonzero_features)

        if len(nonzero_bcs) == 0 or len(nonzero_features) == 0:
            raise NullAxisMatrixError()

        return new_mat, nonzero_bcs, nonzero_features

    def select_nonzero_axes(self) -> Tuple[CountMatrix, np.ndarray, np.ndarray]:
        """Select axes with nonzero sums.

        Returns:
            (CountMatrix, np.array of int, np.array of int):
                New count matrix, non-zero bc indices, feat indices
        """
        return self.select_axes_above_threshold(0)

    def select_nonzero_axis(self, axis: int) -> Tuple[CountMatrix, np.ndarray]:
        """Select axis with nonzero sums.

        Args:
            axis (int): 0 for rows (features) and 1 for columns (barcodes).

        Returns:
            (CountMatrix, np.array of int, np.array of int):
                New count matrix, selected indices
        """

        if axis == 0:
            counts = self.get_counts_per_feature()
            indices = np.flatnonzero(counts > 0)
            return self.select_features(indices), indices

        elif axis == 1:
            counts = self.get_counts_per_bc()
            indices = np.flatnonzero(counts > 0)
            return self.select_barcodes(indices), indices

        else:
            raise ValueError("axis out of range")

    def select_barcodes(self, indices: List[int]) -> CountMatrix:
        """Select a subset of barcodes and return the resulting CountMatrix."""
        return CountMatrix(
            feature_ref=self.feature_ref,
            bcs=[self.bcs[i] for i in indices],
            matrix=self.m[:, indices],
        )

    def select_barcodes_by_seq(self, barcode_seqs: List[bytes]) -> CountMatrix:
        indices = self.bcs_to_ints(barcode_seqs, False)
        return self.select_barcodes(indices)

    def select_barcodes_by_gem_group(self, gem_group: int) -> CountMatrix:
        return self.select_barcodes_by_seq(
            [bc for bc in self.bcs if gem_group == cr_utils.get_gem_group_from_barcode(bc)]
        )

    def select_features(self, indices: Iterable[int]) -> CountMatrix:
        """Select a subset of features and return the resulting matrix.

        We also update FeatureDefs to keep their indices consistent with their new position.
        """
        feature_ref = self.feature_ref.select_features(indices)
        return CountMatrix(feature_ref=feature_ref, bcs=self.bcs, matrix=self.m[indices, :])

    def select_features_by_ids(self, feature_ids: Iterable[bytes]) -> CountMatrix:
        return self.select_features(self.feature_ids_to_ints(feature_ids))

    def remove_genes_not_on_list(self, gene_indices_to_keep: Iterable[int]) -> CountMatrix:
        """Removes all features that are GEX and not on this list, keeping all others.

        Used to subset the matrix down in the targeted assay

        Args:
            gene_indices_to_keep: list of indices of the GEX features to keep

        Returns:
            A copy of this matrix subset to the GEX feature indices requested and all other
            feature types.
        """
        indices = list(gene_indices_to_keep)
        for feature in self.feature_ref.feature_defs:
            if feature.feature_type != rna_library.GENE_EXPRESSION_LIBRARY_TYPE:
                indices.append(feature.index)
        return self.select_features(indices)

    def select_features_by_genome(self, genome: str) -> CountMatrix:
        """Select the subset of gene-expression features for genes in a specific genome"""
        indices = []
        for feature in self.feature_ref.feature_defs:
            if feature.feature_type == rna_library.DEFAULT_LIBRARY_TYPE:
                if feature.tags["genome"] == genome:
                    indices.append(feature.index)
        return self.select_features(indices)

    def select_features_by_types(self, feature_types: Container[str]) -> CountMatrix:
        """Select the subset of gene-expression features by genome and feature type."""
        indices = []
        for feature in self.feature_ref.feature_defs:
            if feature.feature_type in feature_types:
                indices.append(feature.index)
        return self.select_features(indices)

    def select_features_by_type(self, feature_type: str) -> CountMatrix:
        """Select the subset of features with a particular feature type (e.g. "Gene Expression")"""
        indices = []
        for feature in self.feature_ref.feature_defs:
            if feature.feature_type == feature_type:
                indices.append(feature.index)
        return self.select_features(indices)

    def get_feature_ids_by_type(self, feature_type: str) -> List[bytes]:
        """ Return a list of feature ids of a particular feature type (e.g. "Gene Expression")"""
        return self.feature_ref.get_feature_ids_by_type(feature_type)

    def get_count_of_feature_type(self, feature_type: str) -> int:
        """ Count how many features in the matrix are of a given type. (e.g. "Gene Expression")"""
        return self.feature_ref.get_count_of_feature_type(feature_type)

    @staticmethod
    def _get_genomes_from_feature_ref(feature_ref):
        # type: (FeatureReference) -> List[str]
        """Get a list of the distinct genomes represented by gene expression features"""
        return feature_ref.get_genomes(feature_type=rna_library.DEFAULT_LIBRARY_TYPE)

    def get_genomes(self):
        """Get a list of the distinct genomes represented by gene expression features"""
        return CountMatrix._get_genomes_from_feature_ref(self.feature_ref)

    @staticmethod
    def get_genomes_from_h5(filename) -> List[str]:
        """Get a list of the distinct genomes from a matrix HDF5 file"""
        filename = ensure_binary(filename)
        h5_version = CountMatrix.get_format_version_from_h5(filename)
        if h5_version == 1:
            return CountMatrix._get_genomes_from_legacy_v1_h5(filename)
        else:
            with h5.File(filename, "r") as f:
                feature_ref = CountMatrix.load_feature_ref_from_h5_group(f[MATRIX])
                return CountMatrix._get_genomes_from_feature_ref(feature_ref)

    @staticmethod
    def _get_genomes_from_legacy_v1_h5(filename):
        """Get a list of the distinct genomes from a legacy h5py.File (format version 1)"""
        with h5.File(ensure_binary(filename), "r") as f:
            return list(f.keys())

    def get_unique_features_per_bc(self) -> np.ndarray:
        return sum_sparse_matrix(self.m[self.m > 0], axis=0)

    def get_numfeatures_per_bc(self) -> np.ndarray:
        return sum_sparse_matrix(self.m > 0, axis=0)

    def get_counts_per_bc(self) -> np.ndarray:
        return sum_sparse_matrix(self.m, axis=0)

    def get_counts_per_barcode_for_genome(self, genome, feature_type=None) -> np.ndarray:
        """Sum the count matrix across feature rows with a given genome tag.

        The feature reference
        must contain a 'genome' tag. If feature_type is not null filter on it as well.
        """
        assert "genome" in self.feature_ref.all_tag_keys, "feature reference missing 'genome' tag"
        if feature_type:
            indices = [
                i
                for i, fdef in enumerate(self.feature_ref.feature_defs)
                if fdef.tags["genome"] == genome and fdef.feature_type == feature_type
            ]
        else:
            indices = [
                i
                for i, fdef in enumerate(self.feature_ref.feature_defs)
                if fdef.tags["genome"] == genome
            ]
        if indices:
            view = CountMatrixView(self, feature_indices=indices, bc_indices=None)
            return view.sum(axis=0)
        return np.array([], dtype=self.m.dtype)

    def get_counts_per_feature(self) -> np.ndarray:
        return sum_sparse_matrix(self.m, axis=1)

    def get_mean_and_var_per_feature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the mean and variance on the sparse matrix efficiently.

        :return: a tuple with numpy arrays for mean and var
        """
        assert isinstance(self.m, sp_sparse.csc_matrix)
        mean_per_feature = self.m.mean(axis=1)
        second_moment = self.m.copy()
        second_moment = second_moment.power(2.0)
        var_per_feature = second_moment.sum(axis=1) / second_moment.shape[1] - np.power(
            mean_per_feature, 2.0
        )
        var_per_feature = np.asarray(var_per_feature)
        return (mean_per_feature, var_per_feature)

    def get_subselected_counts(
        self, list_feature_ids=None, list_barcodes=None, log_transform=False, library_type=None
    ) -> np.ndarray:
        """Get counts per barcode, sliced various ways

        - subset by list of feature IDs
        - subset by list of barcodes
        - subset by library_type
        """
        subselect_matrix = copy.deepcopy(self)

        if library_type is not None:
            assert (
                library_type in rna_library.RECOGNIZED_FEATURE_TYPES
            ), "library_type not recognized"
            subselect_matrix = subselect_matrix.select_features_by_type(library_type)

        if list_feature_ids is not None:
            subselect_matrix = subselect_matrix.select_features_by_ids(list_feature_ids)

        if list_barcodes is not None:
            subselect_matrix = subselect_matrix.select_barcodes_by_seq(list_barcodes)

        counts_feature = subselect_matrix.get_counts_per_bc()

        if log_transform:
            return np.log10(1.0 + counts_feature)
        return counts_feature

    def get_numbcs_per_feature(self) -> np.ndarray:
        return sum_sparse_matrix(self.m > 0, axis=1)

    def get_top_bcs(self, cutoff) -> np.ndarray:
        reads_per_bc = self.get_counts_per_bc()
        index = max(0, min(reads_per_bc.size, cutoff) - 1)
        value = sorted(reads_per_bc, reverse=True)[index]
        return np.nonzero(reads_per_bc >= value)[0]

    def save_mex(self, base_dir, save_features_func, metadata=None, compress=True):
        """Save in Matrix Market Exchange format.

        Note:
            This operation modifies the matrix by
            converting to a coordinate representation by calling scipy.sparse.csc_matrix.tocoo().

        Args:
          base_dir (str): Path to directory to write files in.
          save_features_func (func): Func that takes (FeatureReference, base_dir, compress) and writes
                                     a file describing the features.
          metadata (dict): Optional metadata to encode into the comments as JSON.
        """
        self.sort_indices()
        self.tocoo()

        cr_io.makedirs(base_dir, allow_existing=True)

        out_matrix_fn = os.path.join(base_dir, "matrix.mtx")
        out_barcodes_fn = os.path.join(base_dir, "barcodes.tsv")
        if compress:
            out_matrix_fn += ".gz"
            out_barcodes_fn += ".gz"

        # This method only supports an integer matrix.
        assert self.m.dtype in ["uint32", "int32", "uint64", "int64"]
        assert isinstance(self.m, sp_sparse.coo.coo_matrix)

        rows, cols = self.m.shape
        # Header fields in the file
        rep = "coordinate"
        field = "integer"
        symmetry = "general"

        metadata = metadata or {}
        metadata.update({"format_version": MATRIX_H5_VERSION})

        metadata_str = tk_safe_json.safe_jsonify(metadata)
        comment = b"metadata_json: %s" % ensure_binary(metadata_str)

        with cr_io.open_maybe_gzip(out_matrix_fn, "wb") as stream:
            # write initial header line
            stream.write(
                np.compat.asbytes(
                    "%%MatrixMarket matrix {0} {1} {2}\n".format(rep, field, symmetry)
                )
            )

            # write comments
            for line in comment.split(b"\n"):
                stream.write(b"%")
                stream.write(ensure_binary(line))
                stream.write(b"\n")

            # write shape spec
            stream.write(np.compat.asbytes("%i %i %i\n" % (rows, cols, self.m.nnz)))
            # write row, col, val in 1-based indexing
            for r, c, d in zip(self.m.row + 1, self.m.col + 1, self.m.data):
                stream.write(np.compat.asbytes(("%i %i %i\n" % (r, c, d))))

        # both GEX and ATAC provide an implementation of this in respective feature_ref.py
        save_features_func(self.feature_ref, base_dir, compress=compress)

        with cr_io.open_maybe_gzip(out_barcodes_fn, "wb") as f:
            for bc in self.bcs:
                f.write(bc + b"\n")

    @staticmethod
    def _get_format_version_from_handle(ofile):
        # type: (h5.File) -> int
        if MATRIX_H5_VERSION_KEY in ofile.attrs:
            version = ofile.attrs[MATRIX_H5_VERSION_KEY]
        else:
            version = 1
        return version

    @staticmethod
    def get_format_version_from_h5(filename):
        with h5.File(ensure_binary(filename), "r") as f:
            return CountMatrix._get_format_version_from_handle(f)

    @staticmethod
    def load_h5_file(filename):
        if isinstance(filename, pathlib.PosixPath):
            fn = filename
        else:
            fn = ensure_binary(filename)

        with h5.File(fn, "r") as f:
            if (
                h5_constants.H5_FILETYPE_KEY not in f.attrs
                or ensure_text(f.attrs[h5_constants.H5_FILETYPE_KEY]) != MATRIX_H5_FILETYPE
            ):
                raise ValueError("HDF5 file is not a valid matrix HDF5 file.")
            version = CountMatrix._get_format_version_from_handle(f)
            if version > MATRIX_H5_VERSION:
                raise ValueError(
                    "Matrix HDF5 file format version (%d) is a newer version that is not supported by this version of the software."
                    % version
                )
            if version < MATRIX_H5_VERSION:
                # raise ValueError('Matrix HDF5 file format version (%d) is an older version that is no longer supported.' % version)
                return CountMatrix.from_legacy_v1_h5(f)

            if MATRIX not in f.keys():
                raise ValueError('Could not find the "matrix" group inside the matrix HDF5 file.')

            return CountMatrix.load(f[MATRIX])

    @staticmethod
    def count_cells_from_h5(filename):
        # NOTE - this double-counts doublets.
        _, bcs, _ = CountMatrix.load_dims_from_h5(filename)
        return bcs

    @staticmethod
    def load_chemistry_from_h5(filename):
        with tables.open_file(ensure_binary(filename), "r") as f:
            try:
                chemistry = f.get_node_attr("/", h5_constants.H5_CHEMISTRY_DESC_KEY)
            except AttributeError:
                chemistry = "Unknown"
        return chemistry

    def filter_barcodes(self, bcs_per_genome):
        # type: (Dict[Any, Iterable[bytes]]) -> CountMatrix
        """Return CountMatrix containing only the specified barcodes.

        Args:
            bcs_per_genome (dict of str to list): Maps genome to cell-associated barcodes.

        Returns:
            CountMatrix w/ the specified barcodes.
        """

        # Union all the cell-associated barcodes
        bcs = set()
        for x in bcs_per_genome.values():
            bcs |= x
        bcs = list(sorted(bcs))
        return self.select_barcodes_by_seq(bcs)

    @staticmethod
    def h5_path(base_path):
        return os.path.join(base_path, "hdf5", "matrices.hdf5")


def merge_matrices(h5_filenames):
    matrix = None
    for h5_filename in h5_filenames:
        if matrix is None:
            matrix = CountMatrix.load_h5_file(h5_filename)
        else:
            other = CountMatrix.load_h5_file(h5_filename)
            matrix.merge(other)
    if matrix is not None:
        matrix.tocsc()
    return matrix


def make_matrix_attrs_count(sample_id, gem_groups, chemistry):
    matrix_attrs = make_library_map_count(sample_id, gem_groups)
    matrix_attrs[h5_constants.H5_CHEMISTRY_DESC_KEY] = chemistry
    return matrix_attrs


def load_matrix_h5_metadata(filename):
    """Get matrix metadata attributes from an HDF5 file"""
    # TODO: Consider moving these to the MATRIX key instead of the root group
    filename = ensure_binary(filename)
    h5_version = CountMatrix.get_format_version_from_h5(filename)
    if h5_version == 1:
        return _load_matrix_legacy_v1_h5_metadata(filename)
    else:
        attrs = {}
        with h5.File(filename, "r") as f:
            for key in h5_constants.H5_METADATA_ATTRS:
                val = f.attrs.get(key)
                if val is not None:
                    if np.isscalar(val) and hasattr(val, "item"):
                        # Coerce numpy scalars to python types.
                        # In particular, force np.unicode_ to unicode (python2)
                        #    or str (python 3)
                        # pylint: disable=no-member
                        attrs[key] = val.item()
                    else:
                        attrs[key] = val
            return attrs


# (needed for compatibility with v1 matrices)
def _load_matrix_legacy_v1_h5_metadata(filename):
    attrs = {}
    with tables.open_file(filename, "r") as f:
        all_attrs = f.get_node("/")._v_attrs
        for key in h5_constants.H5_METADATA_ATTRS:
            if hasattr(all_attrs, key):
                val = getattr(all_attrs, key)
                if np.isscalar(val) and hasattr(val, "item"):
                    # Coerce numpy scalars to python types.
                    # In particular, force np.unicode_ to unicode (python2)
                    #    or str (python 3)
                    attrs[key] = val.item()
                else:
                    attrs[key] = val
    return attrs


def load_matrix_h5_custom_attrs(filename):
    """Get matrix metadata attributes from an HDF5 file"""
    filename = ensure_binary(filename)
    h5_version = CountMatrix.get_format_version_from_h5(filename)
    if h5_version == 1:
        # no support for custom attrs in older versions
        return {}

    attrs = {}
    with h5.File(filename, "r") as f:
        for key, val in f.attrs.items():
            if key not in MATRIX_H5_BUILTIN_ATTRS:
                attrs[key] = val
        return attrs


def make_library_map_count(sample_id, gem_groups):
    # store gem group mapping for use by Cell Loupe
    unique_gem_groups = sorted(set(gem_groups))
    library_map = {
        h5_constants.H5_LIBRARY_ID_MAPPING_KEY: np.array(
            [sample_id] * len(unique_gem_groups), dtype="S"
        ),
        h5_constants.H5_ORIG_GEM_GROUP_MAPPING_KEY: np.array(unique_gem_groups, dtype=int),
    }
    return library_map


def make_library_map_aggr(gem_group_index) -> Dict[str, np.ndarray]:
    # store gem group mapping for use by Cell Loupe
    library_ids = []
    original_gem_groups = []
    # Sort numerically by new gem group
    for _, (lid, og) in sorted(gem_group_index.items(), key=lambda pair: int(pair[0])):
        library_ids.append(lid)
        original_gem_groups.append(og)
    library_map = {
        h5_constants.H5_LIBRARY_ID_MAPPING_KEY: np.array(library_ids, dtype="S"),
        h5_constants.H5_ORIG_GEM_GROUP_MAPPING_KEY: np.array(original_gem_groups, dtype=int),
    }
    return library_map


def get_gem_group_index(matrix_h5):
    with tables.open_file(matrix_h5, mode="r") as f:
        try:
            library_ids = f.get_node_attr("/", h5_constants.H5_LIBRARY_ID_MAPPING_KEY)
            original_gem_groups = f.get_node_attr("/", h5_constants.H5_ORIG_GEM_GROUP_MAPPING_KEY)
        except AttributeError:
            return None
    library_map = {}
    for ng, (lid, og) in enumerate(zip(library_ids, original_gem_groups), start=1):
        library_map[ng] = (lid, og)
    return library_map


def inplace_csc_column_normalize_l2(X):
    """Perform in-place column L2-normalization of input matrix X

    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from sklearn.preprocessing import normalize
    >>> a = np.arange(12, dtype='float').reshape((3, 4))
    >>> b = sp.csc_matrix(a)
    >>> inplace_csc_column_normalize_l2(b)
    >>> np.all(normalize(a, axis=0) == b)
    True
    """
    assert X.getnnz() == 0 or isinstance(X.data[0], (np.float32, float))
    for i in range(X.shape[1]):
        s = 0.0
        for j in range(X.indptr[i], X.indptr[i + 1]):
            s += X.data[j] * X.data[j]
        if s == 0.0:
            continue
        s = np.sqrt(s)
        for j in range(X.indptr[i], X.indptr[i + 1]):
            X.data[j] /= s
