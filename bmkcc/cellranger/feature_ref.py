#!/usr/bin/env python
#
# Copyright (c) 2017 10X Genomics, Inc. All rights reserved.
#

"""Types for loading, saving, and using feature reference data."""

from __future__ import absolute_import, annotations

import os
from collections import OrderedDict

from typing import (
    Collection,
    Dict,
    Generator,
    ItemsView,
    KeysView,
    List,
    Iterable,
    Optional,
    Tuple,
    NamedTuple,
    Union,
    Set,
)

import h5py
import pandas as pd
from six import ensure_binary, ensure_str

import bmkcc.cellranger.hdf5 as cr_h5
import bmkcc.cellranger.h5_constants as h5_constants

FeatureDef = NamedTuple(  # pylint: disable=invalid-name
    "FeatureDef",
    [
        ("index", int),
        ("id", bytes),
        ("name", Optional[str]),
        ("feature_type", str),
        ("tags", Dict[str, str]),
    ],
)

# Required HDF5 datasets
REQUIRED_DATASETS = ["id", "name", "feature_type"]

# These feature tag keys are reserved for internal use.
GENOME_FEATURE_TAG = "genome"
DEFAULT_FEATURE_TAGS = [GENOME_FEATURE_TAG]
RESERVED_TAGS = DEFAULT_FEATURE_TAGS


class FeatureDefException(Exception):
    """Exception type for trying to create a `FeatureReference` with non-distinct IDs."""

    def __init__(self, msg):
        super(FeatureDefException, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class TargetFeatureSets(object):
    """A set of target features.

    Each target set should be a list of integers with a dictionary key representing the name.
    This name will also exist in the library_info.
    """

    def __init__(self, target_features):
        # type: (Union[None, TargetFeatureSets, Dict[str, Iterable[int]]]) -> None
        self.target_feature_sets = dict()  # type: Dict[str, Set[int]]
        if target_features is not None:
            for name, _set in target_features.items():
                assert all(isinstance(x, int) for x in _set)
                _set = set(_set)
                self.target_feature_sets[name] = _set

    def __eq__(self, other):
        if not isinstance(other, TargetFeatureSets):
            return False
        return self.target_feature_sets == other.target_feature_sets

    def __hash__(self):
        return hash(self.target_feature_sets)

    @property
    def names(self) -> KeysView[str]:
        return self.target_feature_sets.keys()

    def items(self) -> ItemsView[str, Set[int]]:
        return self.target_feature_sets.items()

    def iteritems(self) -> ItemsView[str, Set[int]]:
        """Returns a view of the items.

        .. deprecated::4.0
            Use `items` instead.  Python 3 doesn't do `iteritems`.
        """
        return self.target_feature_sets.items()

    def get(self, key: str) -> Optional[Set[int]]:
        return self.target_feature_sets.get(key)

    def update(self, other: Dict[str, Set[int]]) -> None:
        self.target_feature_sets.update(other)

    def get_target_feature_indices(self) -> List[int]:
        return sorted(set.union(*self.target_feature_sets.values()))

    def subset_to_reduced_features(
        self, old_to_new_index_translation: Dict[int, int]
    ) -> TargetFeatureSets:
        """Recreates a list of target features by subsetting it.

        If the Feature Reference holding this object is ever subset, or the index values are changed
        we need to update the target_feature_sets to hold these new values instead.  This function
        recreates a list of target features by subsetting it to only those present in a translation
        table that maps old index positions to new ones.

        Args:
            old_to_new_index_translation: a dictionary that tells what old index position is being converted
            to a new index position

        Returns:
            A new TargetFeatureSets
        """
        assert isinstance(old_to_new_index_translation, dict)
        new_target_features = {}
        for name, indices in self.target_feature_sets.items():
            new_vals = [
                old_to_new_index_translation[old_index]
                for old_index in indices
                if old_index in old_to_new_index_translation
            ]
            new_target_features[name] = new_vals
        return TargetFeatureSets(new_target_features)


class FeatureReference(object):
    """Store a list of features (genes, antibodies, etc)."""

    def __init__(self, feature_defs, all_tag_keys, target_features=None):
        # type: (List[FeatureDef], List[str], Union[None, TargetFeatureSets, Dict[str, Iterable[int]]]) -> None
        """Create a FeatureReference.

        Args:
            feature_defs (list of FeatureDef): All feature definitions.
            all_tag_keys (list of str): All optional tag keys.
            target_features (dictionary of list of int): Optional target set(s). Each target set
            should be a list of integers with a dictionary key representing the name. This name
            will also exist in the library_info.
        """
        self.feature_defs = feature_defs
        self.all_tag_keys = all_tag_keys

        if target_features is not None:
            self.target_features = TargetFeatureSets(target_features)
        else:
            self.target_features = None

        # Assert uniqueness of feature IDs
        id_map = {}
        for fdef in self.feature_defs:
            if fdef.id in id_map:
                this_fd_str = "ID: %s; name: %s; type: %s" % (fdef.id, fdef.name, fdef.feature_type)
                seen_fd_str = "ID: %s; name: %s; type: %s" % (
                    id_map[fdef.id].id,
                    id_map[fdef.id].name,
                    id_map[fdef.id].feature_type,
                )
                raise FeatureDefException(
                    "Found two feature definitions with the same ID: "
                    "(%s) and (%s). All feature IDs must be distinct." % (this_fd_str, seen_fd_str)
                )
            id_map[fdef.id] = fdef

        self.id_map = id_map  # type: Dict[bytes, FeatureDef]

    def get_count_of_feature_type(self, feature_type: str) -> int:
        """ Count how many features in the matrix are of a given type. (e.g. "Gene Expression")"""
        total = 0
        for feature in self.feature_defs:
            if feature.feature_type == feature_type:
                total += 1
        return total

    def __eq__(self, other):
        return (
            self.feature_defs == other.feature_defs
            and self.all_tag_keys == other.all_tag_keys
            and (self.target_features is None) == (other.target_features is None)
            and self.target_features == other.target_features
        )

    def __hash__(self):
        if self.target_features is None:
            return hash((self.feature_defs, self.all_tag_keys))
        return hash((self.feature_defs, self.all_tag_keys, self.target_features))

    def __ne__(self, other):
        return not self == other

    def equals_ignore_target_set(self, other: FeatureReference) -> bool:
        """Checks if two feature references are equal, ignoring missing targets.

        Only takes into account equality of the
        target set if it is present in both feature references. Useful for targeted feature reference
        compatibility checks in preflights.
        """
        ignore_target_set = (not self.has_target_features()) or (not other.has_target_features())
        compatible_target_sets = ignore_target_set or (
            self.get_target_feature_indices() == other.get_target_feature_indices()
        )
        return (
            self.feature_defs == other.feature_defs
            and self.all_tag_keys == other.all_tag_keys
            and compatible_target_sets
        )

    @staticmethod
    def addtags(
        feature_ref: FeatureReference,
        new_tags: List[str],
        new_labels: Optional[Collection[Collection[str]]] = None,
    ) -> FeatureReference:
        """Add new tags and corresponding labels to existing feature_ref.

        If new labels are None, empty strings are supplied by default

        Args:
            feature_ref: a FeatureReference instance
            new_tags: a list of new tags
            new_labels: per feature list of label values corresponding to the new tags
        """
        assert len(new_tags) > 0
        for tag in new_tags:
            if tag in feature_ref.all_tag_keys:
                raise ValueError("tag {} is already present in feature_ref")

        if new_labels is not None:
            if len(feature_ref.feature_defs) != len(new_labels):
                raise ValueError(
                    "number of labels does not match number of features " "in feature_ref"
                )
            for labels in new_labels:
                assert len(labels) == len(new_tags)
            use_labels = new_labels
        else:
            # initialize to empty
            use_labels = [[""] * len(new_tags) for _ in range(len(feature_ref.feature_defs))]
        assert len(feature_ref.feature_defs) == len(use_labels)

        augmented_features = []
        for f_def, newvals in zip(feature_ref.feature_defs, use_labels):
            tags = dict(zip(new_tags, newvals))
            tags.update(f_def.tags)
            augmented_features.append(
                FeatureDef(
                    index=f_def.index,
                    id=f_def.id,
                    name=f_def.name,
                    feature_type=f_def.feature_type,
                    tags=tags,
                )
            )

        return FeatureReference(
            feature_defs=augmented_features,
            all_tag_keys=feature_ref.all_tag_keys + new_tags,
            target_features=feature_ref.target_features,
        )

    @staticmethod
    def join(feature_ref1: FeatureReference, feature_ref2: FeatureReference) -> FeatureReference:
        """Concatenate two feature references, requires unique ids and identical tags"""
        assert feature_ref1.all_tag_keys == feature_ref2.all_tag_keys
        feature_defs1 = feature_ref1.feature_defs
        feature_defs2 = feature_ref2.feature_defs

        if feature_ref1.target_features is None:
            combined_target_features = feature_ref2.target_features
        elif feature_ref2.target_features is None:
            combined_target_features = feature_ref1.target_features
        else:
            combined_target_features = feature_ref1.target_features
            # if feature_ref2 has the same keys, they will be over-written
            combined_target_features.update(feature_ref2.target_features)
        return FeatureReference(
            feature_defs=feature_defs1 + feature_defs2,
            all_tag_keys=feature_ref1.all_tag_keys,
            target_features=combined_target_features,
        )

    @classmethod
    def empty(cls) -> FeatureReference:
        return cls(feature_defs=[], all_tag_keys=[], target_features=None)

    def get_num_features(self) -> int:
        return len(self.feature_defs)

    def get_feature_ids_by_type(self, feature_type: str) -> List[bytes]:
        """ Return a list of feature ids of a particular feature type (e.g. "Gene Expression")"""
        return [f.id for f in self.feature_defs if f.feature_type == feature_type]

    def get_indices_for_type(self, feature_type: str) -> List[int]:
        return [
            feature.index for feature in self.feature_defs if feature.feature_type == feature_type
        ]

    def get_genomes(self, feature_type: Optional[str] = None) -> List[str]:
        """Get sorted list of genomes. Empty string is for reverse compatibility.

        A specific feature type can optionally be specified.
        """
        genomes = set(
            f.tags.get(GENOME_FEATURE_TAG, "")
            for f in self.feature_defs
            if (feature_type is None or f.feature_type == feature_type)
        )
        return sorted(genomes)

    def has_target_features(self) -> bool:
        return self.target_features is not None

    def get_target_feature_indices(self) -> Optional[List[int]]:
        """Gets the indices of on-target features within the FeatureRerence.

        Returns None if there is no target set.
        """
        if not self.has_target_features():
            return None
        else:
            return self.target_features.get_target_feature_indices()

    def get_target_feature_ids(self) -> Optional[List[bytes]]:
        """Gets the feature ids of on-target features.

        Returns None if there is no target set.
        """
        if not self.has_target_features():
            return None
        else:
            return [self.feature_defs[i].id for i in self.get_target_feature_indices()]

    def select_features_by_type(self, feature_type: str) -> FeatureReference:
        indices = [i for i, fd in enumerate(self.feature_defs) if fd.feature_type == feature_type]
        return self.select_features(indices)

    def get_feature_names(self) -> List[str]:
        """Get a list of feature names.

        :return: [x.name for x in self.feature_defs]
        """
        return [x.name for x in self.feature_defs]

    def select_features(self, indices: Iterable[int]) -> FeatureReference:
        """Create a new FeatureReference that only contains features with the given indices.

        Any target sets present are updated, but the target_features attribute in the output
        is set to None if all target features were removed.
        """
        old_defs = [self.feature_defs[i] for i in indices]
        new_defs = []  # type: List[FeatureDef]
        translation_map = {}
        for i, f_def in enumerate(old_defs):
            translation_map[f_def.index] = i
            new_defs.append(
                FeatureDef(
                    index=i,
                    id=f_def.id,
                    name=f_def.name,
                    feature_type=f_def.feature_type,
                    tags=f_def.tags,
                )
            )

        if self.has_target_features():
            new_target_features = self.target_features.subset_to_reduced_features(translation_map)
            # If we have removed all targeted features, we toss the empty target sets
            if len(new_target_features.get_target_feature_indices()) == 0:
                new_target_features = None
        else:
            new_target_features = None
        return FeatureReference(
            feature_defs=new_defs,
            all_tag_keys=self.all_tag_keys,
            target_features=new_target_features,
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write to an HDF5 group."""

        # Write required datasets
        for col in REQUIRED_DATASETS:
            data = [getattr(f, col) for f in self.feature_defs]
            cr_h5.create_hdf5_string_dataset(group, col, data, compression=True)

        # Write tag datasets
        for col in self.all_tag_keys:
            # Serialize missing data as empty unicode string
            data = [f.tags.get(col, "") for f in self.feature_defs]
            cr_h5.create_hdf5_string_dataset(group, col, data, compression=True)

        # Write target_features as a new sub-group
        if self.target_features is not None:
            target_set_group = group.create_group(h5_constants.H5_TARGET_SET_ATTR)
            for key, val in self.target_features.items():
                cr_h5.create_hdf5_string_dataset(
                    target_set_group, key, [str(x) for x in sorted(val)], compression=True
                )

        # Record names of all tag columns
        cr_h5.create_hdf5_string_dataset(group, "_all_tag_keys", self.all_tag_keys)

    @classmethod
    def from_hdf5(cls, group: h5py.Dataset) -> FeatureReference:
        """Load from an HDF5 group.

        Args:
            group (h5py.Dataset): Group to load from.

        Returns:
            feature_ref (FeatureReference): New object.
        """
        # FIXME: ordering may not be guaranteed in python3
        def _load_str(node: h5py.Dataset) -> List[str]:
            if node.shape is None:
                return []
            if node.dtype.char == cr_h5.STR_DTYPE_CHAR:
                memoize = node.name in ("/features/feature_type", "/features/genome")
                return cr_h5.read_hdf5_string_dataset(node, memoize)
            else:
                return node[:]

        def _format_path(path) -> str:
            """ Strip off leading slash and convert to str (could be unicode). """
            return ensure_str(path[1:])

        def _h5py_dataset_iterator(
            group: h5py.Group, prefix: str = ""
        ) -> Generator[Tuple[str, List[str]], None, None]:
            for key in group:
                item = group[key]
                path = "/".join([prefix, key])
                if isinstance(item, h5py.Dataset):
                    yield _format_path(path), _load_str(item)
                elif isinstance(item, h5py.Group):
                    # TODO: in python 3.3 this can be a yield from statement
                    for subpath, subitem in _h5py_dataset_iterator(item, path):
                        yield subpath, subitem

        data = OrderedDict(_h5py_dataset_iterator(group))

        # Load Tag Keys
        all_tag_keys = data["_all_tag_keys"]

        # Load Target Sets, if they exist
        target_features = {
            os.path.basename(key): [int(x) for x in val]
            for key, val in data.items()
            if h5_constants.H5_TARGET_SET_ATTR in ensure_binary(key)
        }
        if len(target_features) == 0:
            target_features = None

        # Build FeatureDefs
        feature_defs = []  # List[FeatureDef]
        num_features = len(data[REQUIRED_DATASETS[0]])

        for i in range(num_features):
            tags = {ensure_str(k): ensure_str(data[k][i]) for k in all_tag_keys if data[k][i]}
            feature_defs.append(
                FeatureDef(
                    id=ensure_binary(data["id"][i]),
                    index=i,
                    name=data["name"][i],
                    feature_type=data["feature_type"][i],
                    tags=tags,
                )
            )

        return cls(feature_defs, all_tag_keys=all_tag_keys, target_features=target_features)


class SummaryFeatureCounts(object):
    """Stores and queries a dataframe that summarizes relevant features per gene."""

    def __init__(self, df: pd.DataFrame, filepath=None):
        """Stores and queries a dataframe that summarizes relevant features per gene.

        There should be one row per gene and any number of columns.

        Args:
            df (pandas dataframe): Dataframe with n_genes rows and columns with features per genes.
        """
        self.df = df
        # for easy querying
        self.feature_dict = df.set_index("feature_id").to_dict(orient="index")
        self.filepath = filepath

    @classmethod
    def from_file(cls, path) -> SummaryFeatureCounts:
        """Loads the features from a csv file."""
        path = ensure_str(path)
        assert os.path.isfile(path)
        return cls(pd.read_csv(ensure_str(path)), path)

    def to_csv(self, path):
        path = ensure_str(path)
        self.df.to_csv(path, index=False)

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_value_for_feature(self, feature_id, key):
        """Get the given key for the given feature."""
        if feature_id not in self.feature_dict:
            return None
        if key not in self.feature_dict[feature_id]:
            return None
        return self.feature_dict[feature_id][key]
