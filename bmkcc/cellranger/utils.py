#!/usr/bin/env python
#
# Copyright (c) 2015 10X Genomics, Inc. All rights reserved.
#

"""
Assorted grab-bag of miscellaneous helper methods.

Do not add to this module.  Instead, find or create a module with a name
that indicates to a potential user what sorts of methods they might find
in that module.
"""

from __future__ import absolute_import, annotations, division

import collections
import csv
import itertools
import json
import os
import random
import h5py
import numpy as np
from six import ensure_binary, ensure_text
import bmkcc.tenkit.log_subprocess as tk_subproc
import bmkcc.tenkit.seq as tk_seq
import bmkcc.tenkit.constants as tk_constants
import bmkcc.cellranger.constants as cr_constants
import bmkcc.cellranger.h5_constants as h5_constants

from typing import (  # pylint: disable=unused-import
    AnyStr,
    Dict,
    Generator,
    IO,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from pysam import AlignmentFile, AlignedSegment  # pylint: disable=unused-import


def get_gem_group_from_barcode(barcode: Optional[bytes]) -> Optional[int]:
    """
    Method to get just the gem group from a barcode

    Args:
        barcode (bytes): a barcode, a la "ACGTACTAGAC-1"

    Returns:
        The Gem group as an int or None if not present.
    """
    if barcode is None:
        return None
    gg = barcode.index(b"-")
    if gg > 0:
        return int(barcode[gg + 1 :])
    else:
        return None


def _load_reference_metadata_file(reference_path: str,) -> Dict[str, Union[dict, list, str, int]]:
    reference_metadata_file = os.path.join(reference_path, cr_constants.REFERENCE_METADATA_FILE)
    with open(reference_metadata_file, "r") as f:
        return json.load(f)


def get_reference_star_path(reference_path: str) -> str:
    return os.path.join(reference_path, cr_constants.REFERENCE_STAR_PATH)


def get_reference_genome_fasta(reference_path: str) -> str:
    return os.path.join(reference_path, cr_constants.REFERENCE_FASTA_PATH)


def get_reference_genomes(reference_path: str) -> List[str]:
    data = _load_reference_metadata_file(reference_path)
    return data[cr_constants.REFERENCE_GENOMES_KEY]


def is_arc_reference(reference_path: str) -> bool:
    data = _load_reference_metadata_file(reference_path)
    return data.get("mkref_version", "").startswith("cellranger-arc")


def get_reference_mem_gb_request(reference_path: str) -> int:
    data = _load_reference_metadata_file(reference_path)
    return data[cr_constants.REFERENCE_MEM_GB_KEY]


def barcode_sort_key(read, squash_unbarcoded: bool = False) -> Optional[tuple]:
    formatted_bc = get_read_barcode(read)
    if squash_unbarcoded and formatted_bc is None:
        return None
    (bc, gg) = split_barcode_seq(formatted_bc)
    library_idx = get_read_library_index(read)
    return gg, bc, library_idx, get_read_raw_umi(read)


def pos_sort_key(read: AlignedSegment) -> tuple:
    return read.tid, read.pos


def format_barcode_seq(barcode: bytes, gem_group: Optional[int] = None) -> bytes:
    if gem_group is not None:
        barcode += b"-%d" % gem_group
    return barcode


def format_barcode_seqs(
    barcode_seqs: List[bytes], gem_groups: Optional[Iterable[int]]
) -> List[bytes]:
    if gem_groups is None:
        return barcode_seqs
    new_barcode_seqs = []
    unique_gem_groups = sorted(set(gem_groups))
    for gg in unique_gem_groups:
        new_barcode_seqs += [format_barcode_seq(bc, gg) for bc in barcode_seqs]
    return new_barcode_seqs


def split_barcode_seq(
    barcode: Optional[bytes],
) -> Union[Tuple[None, None], Tuple[bytes, Optional[int]]]:
    if barcode is None:
        return None, None

    barcode_parts = barcode.split(b"-")

    barcode = barcode_parts[0]
    if len(barcode_parts) > 1:
        gem_group = int(barcode_parts[1])
    else:
        gem_group = None

    return barcode, gem_group


def bcs_suffices_to_names(bcs: Iterable[bytes], gg_id_map: Dict[int, AnyStr]) -> List[AnyStr]:
    """Turn list of barcodes into corresponding list of mapped values.

    Args:
        bcs (list): aggr barcodes with suffix corresponding to gem group id
        gg_id_map (dict): mapping gem-group-ids/barcode-suffixes (int) to
            a desired named value (typically `aggr_id` for the library id of that
            one sample, or `batch_id` for the name of a batch it is a part of.)
    """
    mapped_ids = [gg_id_map[split_barcode_seq(bc)[1]] for bc in bcs]
    return mapped_ids


def is_barcode_corrected(raw_bc_seq: bytes, processed_bc_seq: Optional[bytes]) -> bool:
    if processed_bc_seq is None:
        return False

    bc_seq, _ = split_barcode_seq(processed_bc_seq)
    return bc_seq != raw_bc_seq


def get_genome_from_str(s, genomes):
    assert len(genomes) > 0

    if s is None:
        return None

    if len(genomes) == 1:
        return genomes[0]

    for genome in genomes:
        if s.startswith(genome):
            return genome

    raise Exception("%s does not have valid associated genome" % s)


def remove_genome_from_str(s, genomes, prefixes_as_genomes=False):
    assert len(genomes) > 0
    if s is None:
        return None
    if len(genomes) == 1:
        return s

    # Gene names/ids/chroms are padded with N underscores to achieve the same prefix length
    #   for all genomes, e.g., GRCh38_* and mm10___*
    max_len = max(len(g) for g in genomes)

    for genome in genomes:
        if s.startswith(genome):
            # Strip genome and subsequent underscores
            if prefixes_as_genomes:
                return s[(1 + len(genome)) :]
            else:
                return s[(1 + max_len) :]

    raise Exception("%s does not have valid associated genome" % s)


def get_genome_from_read(read, chroms, genomes):
    assert len(genomes) > 0

    if read.is_unmapped:
        return None

    if len(genomes) == 1:
        return genomes[0]

    return get_genome_from_str(chroms[read.tid], genomes)


def get_read_barcode(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.PROCESSED_BARCODE_TAG)


def get_read_raw_barcode(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.RAW_BARCODE_TAG)


def get_read_barcode_qual(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.RAW_BARCODE_QUAL_TAG)


def get_read_umi_qual(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.UMI_QUAL_TAG)


def get_read_umi(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.PROCESSED_UMI_TAG)


def get_read_raw_umi(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.RAW_UMI_TAG)


def get_read_library_index(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.LIBRARY_INDEX_TAG)


def get_read_gene_ids(read: AlignedSegment):
    s = _get_read_tag(read, cr_constants.FEATURE_IDS_TAG)  # type: Optional[str]
    if s is None:
        return None
    assert isinstance(s, str)
    return tuple(s.split(";"))


def get_read_transcripts_iter(read: AlignedSegment):
    """Iterate over all transcripts compatible with the given read.

    We do this by iterating over the TX tag entries that are of the form
    `TX:<transcript id>,<strand><position>,<cigar>`.

    Note:
        When intronic alignment is turned on, the `TX` tag is set to
        `TX:<gene id>,<strand>` for intronic  reads or exonic reads that are
        not compatible with annotated splice junctions. We ignore these.
    """
    s = _get_read_tag(read, cr_constants.TRANSCRIPTS_TAG)
    if s is None:
        return

    s = ensure_binary(s)
    for x in s.split(b";"):
        if len(x) == 0:
            continue

        parts = x.split(b",")
        # len(parts) = 3 for transcriptomic alignments, while for intronic alignments when
        # intron counting mode is enabled len(parts)=2
        if len(parts) != 3:
            continue

        chrom = parts[0] if parts[0] else None
        if parts[1]:
            strand = parts[1][0:1]
            pos = int(parts[1][1:])
        else:
            strand = None
            pos = None
        cigarstring = parts[2] if parts[2] else None
        assert strand in cr_constants.STRANDS or strand is None

        yield chrom, strand, pos, cigarstring


REGION_TAG_MAP = {
    "E": cr_constants.EXONIC_REGION,
    "N": cr_constants.INTRONIC_REGION,
    "I": cr_constants.INTERGENIC_REGION,
}


def get_mapping_region(read: AlignedSegment) -> Optional[str]:
    region_tag = _get_read_tag(read, "RE")
    return REGION_TAG_MAP.get(region_tag, None)


def iter_by_qname(in_genome_bam, in_trimmed_bam):
    """Iterate through multiple BAMs by qname simultaneously.

    Assume the trimmed-read-bam has every qname in the genome bam, in the same order.
    """

    genome_bam_iter = itertools.groupby(in_genome_bam, key=lambda read: read.qname)

    if in_trimmed_bam is None:
        trimmed_bam_iter = iter(())
    else:
        trimmed_bam_iter = itertools.groupby(in_trimmed_bam, key=lambda read: read.qname)

    for (genome_qname, genome_reads), trimmed_tuple in itertools.zip_longest(
        genome_bam_iter, trimmed_bam_iter
    ):
        trimmed_qname, trimmed_reads = trimmed_tuple or (None, [])
        genome_reads = list(genome_reads)
        trimmed_reads = list(trimmed_reads)

        assert (in_trimmed_bam is None) or trimmed_qname == genome_qname
        yield (genome_qname, genome_reads, trimmed_reads)


def set_read_tags(read: AlignedSegment, tags) -> None:
    read_tags = read.tags
    read_tags.extend(tags)
    read.tags = read_tags


def _get_read_tag(read: AlignedSegment, tag):
    try:
        r = read.opt(tag)
        if not r:
            r = None
        return r
    except KeyError:
        return None


def get_fastq_read1(r1, r2, reads_interleaved):
    if reads_interleaved:
        name1, seq1, qual1, _, _, _ = r1
    else:
        name1, seq1, qual1 = r1
    return name1, seq1, qual1


def get_fastq_read2(r1, r2, reads_interleaved):
    if reads_interleaved:
        _, _, _, name2, seq2, qual2 = r1
    else:
        name2, seq2, qual2 = r2
    return name2, seq2, qual2


def get_read_extra_flags(read: AlignedSegment):
    return _get_read_tag(read, cr_constants.EXTRA_FLAGS_TAG) or 0


# EXTRA_FLAGS_CONF_MAPPED_TXOME = 1
EXTRA_FLAGS_LOW_SUPPORT_UMI = 2
# EXTRA_FLAGS_GENE_DISCORDANT = 4
EXTRA_FLAGS_UMI_COUNT = 8
EXTRA_FLAGS_CONF_MAPPED_FEATURE = 16
EXTRA_FLAGS_FILTERED_TARGET_UMI = 32


def is_read_low_support_umi(read: AlignedSegment) -> bool:
    return (get_read_extra_flags(read) & EXTRA_FLAGS_LOW_SUPPORT_UMI) > 0


def is_read_filtered_target_umi(read: AlignedSegment) -> bool:
    return (get_read_extra_flags(read) & EXTRA_FLAGS_FILTERED_TARGET_UMI) > 0


def is_read_umi_count(read: AlignedSegment) -> bool:
    return (get_read_extra_flags(read) & EXTRA_FLAGS_UMI_COUNT) > 0


def is_read_conf_mapped_to_feature(read: AlignedSegment) -> bool:
    return (get_read_extra_flags(read) & EXTRA_FLAGS_CONF_MAPPED_FEATURE) > 0


def is_read_dupe_candidate(read, high_conf_mapq, use_corrected_umi=True, use_umis=True) -> bool:
    if use_corrected_umi:
        umi = get_read_umi(read)
    else:
        umi = get_read_raw_umi(read)

    return (
        not read.is_secondary
        and (umi is not None or not use_umis)
        and (get_read_barcode(read) is not None)
        and not is_read_low_support_umi(read)
        and not is_read_filtered_target_umi(read)
        and (
            is_read_conf_mapped_to_transcriptome(read, high_conf_mapq)
            or is_read_conf_mapped_to_feature(read)
        )
    )


def is_read_conf_mapped(read: AlignedSegment, high_conf_mapq) -> bool:
    if read.is_unmapped:
        return False
    elif read.mapq < high_conf_mapq:
        return False
    return True


def is_read_conf_mapped_to_transcriptome(read: AlignedSegment, high_conf_mapq) -> bool:
    if is_read_conf_mapped(read, high_conf_mapq):
        gene_ids = get_read_gene_ids(read)
        return (gene_ids is not None) and (len(gene_ids) == 1)
    return False


def is_read_conf_mapped_to_transcriptome_barcoded(read, high_conf_mapq) -> bool:
    return (
        is_read_conf_mapped_to_transcriptome(read, high_conf_mapq)
        and not read.is_secondary
        and get_read_barcode(read)
    )


def is_read_conf_mapped_to_transcriptome_barcoded_deduped(read, high_conf_mapq) -> bool:
    return (
        is_read_conf_mapped_to_transcriptome_barcoded(read, high_conf_mapq)
        and not read.is_duplicate
    )


def is_read_conf_mapped_to_transcriptome_barcoded_deduped_with_umi(read, high_conf_mapq) -> bool:
    return is_read_conf_mapped_to_transcriptome_barcoded_deduped(
        read, high_conf_mapq
    ) and get_read_umi(read)


def load_barcode_summary(barcode_summary):
    if barcode_summary:
        with h5py.File(barcode_summary) as f:
            return list(f[cr_constants.H5_BC_SEQUENCE_COL])
    return None


def compress_seq(s: bytes, bits: int = 64):
    """Pack a DNA sequence (no Ns!) into a 2-bit format, into a python int.

    Args:
        s (str): A DNA sequence.
    Returns:
        int: The sequence packed into the bits of an integer.
    """
    assert len(s) <= (bits // 2 - 1)
    result = 0
    for i in range(len(s)):
        nuc = s[i : i + 1]
        assert nuc in tk_seq.NUCS_INVERSE
        result = result << 2
        result = result | tk_seq.NUCS_INVERSE[nuc]
    return result


def set_tag(read, key, old_value, new_value):
    """ Set a bam tag for a read, overwriting the previous value """
    tags = read.tags
    tags.remove((key, old_value))
    tags.append((key, new_value))
    read.tags = tags


def build_alignment_param_metrics() -> Dict[str, Union[str, int]]:
    """ hard-coded aligner settings"""

    align = {"aligner": "star", "high_conf_mapq": cr_constants.STAR_DEFAULT_HIGH_CONF_MAPQ}
    aln_param_metrics = {}
    for key, value in align.items():
        aln_param_metrics["alignment_" + key] = value
    return aln_param_metrics


def get_high_conf_mapq() -> int:
    return cr_constants.STAR_DEFAULT_HIGH_CONF_MAPQ


def get_mem_gb_request_from_genome_fasta(reference_path: str) -> float:
    in_fasta_fn = get_reference_genome_fasta(reference_path)
    genome_size_gb = float(os.path.getsize(in_fasta_fn)) / 1e9
    return np.ceil(
        max(
            h5_constants.MIN_MEM_GB,
            cr_constants.BAM_CHUNK_SIZE_GB + max(1, 2 * int(genome_size_gb)),
        )
    )


def merge_jsons_as_dict(in_filenames: Iterable[Union[str, bytes, os.PathLike]]) -> dict:
    """ Merge a list of json files and return the result as a dictionary """
    d = {}
    for filename in in_filenames:
        if (filename is None) or (not (os.path.isfile(filename))):
            continue
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                d.update(data)
        except IOError:
            continue
    return d


def format_barcode_summary_h5_key(
    library_prefix: str, genome: str, region: str, read_type: str
) -> str:
    """Formats the barcode summary into a key to access the `barcode_summary.h5` outs.

    Here, we need to accommodate both accessing old and new versions of the h5 file
    compatible with both v2 and v3 of the matrix.

    Args:
        library_prefix (str): Name of the library, used as a prefix
        genome         (str): Name of the genome used for alignment
        region         (str): Name of the subset of the genome we are looking at (e.g. transcriptome, regulome,
                            epigenome, ...). This should be a controlled vocabulary in `cellranger.constants`
        read_type      (str): Name of the read types we are trying to extract (e.g. `conf_mapped_deduped_barcoded`, ...).
                            It should be also controlled in `cellranger.constants`.
    Returns:
        output_key     (str): A string constant with the suffix `reads` appended.
    """
    # Clean the input for non-found or undefined keys. Will break badly if they are not controlled, but will let us
    # keep compatible across v2 and v3 developments.
    str_list = [library_prefix, genome, region, read_type]
    # Append reads
    str_list.append("reads")
    # join
    output_key = "_".join(str_list)
    return output_key


def downsample(rate: Optional[float]) -> bool:
    if rate is None or rate == 1.0:
        return True
    return random.random() <= rate


def numpy_groupby(values, keys) -> Generator[Tuple[tuple, tuple], None, None]:
    """Group a collection of numpy arrays by key arrays.

    Yields `(key_tuple, view_tuple)` where `key_tuple` is the key grouped
    on and `view_tuple` is a tuple of views into the value arrays.

    Args:
        values (tuple of arrays): tuple of arrays to group.
        keys (tuple): tuple of sorted, numeric arrays to group by.
    Returns:
        sequence of tuple: Sequence of (`key_tuple`, `view_tuple`).
    """

    if len(values) == 0:
        return
    if len(values[0]) == 0:
        return

    for key_array in keys:
        assert len(key_array) == len(keys[0])
    for value_array in values:
        assert len(value_array) == len(keys[0])

    # The indices where any of the keys differ from the previous key become group boundaries
    # pylint: disable=no-member
    key_change_indices = np.logical_or.reduce(
        tuple(np.concatenate(([1], np.diff(key))) != 0 for key in keys)
    )
    group_starts = np.flatnonzero(key_change_indices)
    group_ends = np.roll(group_starts, -1)
    group_ends[-1] = len(keys[0])

    for group_start, group_end in zip(group_starts, group_ends):
        yield tuple(key[group_start] for key in keys), tuple(
            value[group_start:group_end] for value in values
        )


def get_seqs(l: int) -> List[bytes]:
    if l == 1:
        return tk_seq.NUCS
    old_seqs = get_seqs(l - 1)
    new_seqs = []  # type: List[bytes]
    for old_seq in old_seqs:
        for base in tk_seq.NUCS:
            new_seqs.append(old_seq + base)
    return new_seqs


def get_fasta_iter(f: IO[bytes]) -> Generator[Tuple[bytes, bytes], None, None]:
    hdr = b""
    seq = b""
    for line in f:
        line = line.strip()
        if line.startswith(b">"):
            if hdr:
                yield hdr, seq
            hdr = line[1:]
            seq = b""
        else:
            seq += line
    if hdr:
        yield hdr, seq


# Pysam numeric codes to meaningful categories
_cigar_numeric_to_category_map = {
    0: "M",
    1: "I",
    2: "D",
    3: "N",
    4: "S",
    5: "H",
    6: "P",
    7: "=",
    8: "X",
}


def get_cigar_summary_stats(read: AlignedSegment, strand: bytes) -> Dict[str, int]:
    """
    Get number of mismatches, insertions, deletions, ref skip, soft clip, hard clip bases from a read.

    Returns a dictionary by the element's CIGAR designation. Adds additional
    fields to distinguish between three and five prime soft-clipping for `R1`
    and `R2`: `R1_S_three_prime` and `R1_S_five_prime`, etc. to account for
    soft-clipped local alignments.

    Args:
        read (pysam.AlignedRead): aligned read object
        strand (string): + or - to indicate library orientation (MRO argument strand, for example)

    Returns:
        dict of str,int: Key of base type to base counts for metrics. Adds
                         additional fields to distinguish between three and
                         five prime soft-clipping: `S_three_prime` and
                         `S_five_prime`.
    """

    statistics = {}  # Dict[str, int]
    cigar_tuples = read.cigar

    for i, (category, count) in enumerate(cigar_tuples):
        # Convert numeric code to category
        category = _cigar_numeric_to_category_map[category]
        count = int(count)

        # Detect 5 prime soft-clipping
        if i == 0:

            if strand == cr_constants.REVERSE_STRAND:
                metric = "R1_S_five_prime" if read.is_read1 else "R2_S_three_prime"
            else:
                metric = "R2_S_five_prime" if read.is_read1 else "R1_S_three_prime"

            if category == "S":
                statistics[metric] = count
            else:
                statistics[metric] = 0

        # Tally up all standard categories from BAM
        if category in statistics:
            statistics[category] += count
        else:
            statistics[category] = count

        # Detect 3 prime soft-clipping
        if i == len(cigar_tuples):
            if strand == cr_constants.REVERSE_STRAND:
                metric = "R2_S_five_prime" if read.is_read2 else "R1_S_three_prime"
            else:
                metric = "R1_S_five_prime" if read.is_read2 else "R2_S_three_prime"

            if category == "S":
                statistics[metric] = count
            else:
                statistics[metric] = 0

    return statistics


def get_full_alignment_base_quality_scores(read: AlignedSegment) -> Tuple[int, np.ndarray]:
    """
    Returns base quality scores for the full read alignment.

    Inserts zeroes for deletions and removing inserted and soft-clipped bases.
    Therefore, only returns quality for truly aligned sequenced bases.

    Args:
        read (pysam.AlignedSegment): read to get quality scores for

    Returns:
        np.array: numpy array of quality scores
    """

    # pylint: disable=no-member
    quality_scores = np.fromstring(read.qual, dtype=np.byte) - tk_constants.ILLUMINA_QUAL_OFFSET

    start_pos = 0

    for operation, length in read.cigar:
        operation = _cigar_numeric_to_category_map[operation]

        if operation == "D":
            quality_scores = np.insert(quality_scores, start_pos, [0] * length)
        elif operation == "I" or operation == "S":
            quality_scores = np.delete(quality_scores, np.s_[start_pos : start_pos + length])

        if not operation == "I" and not operation == "S":
            start_pos += length

    return start_pos, quality_scores


def get_unmapped_read_count_from_indexed_bam(bam_file_name: AnyStr) -> int:
    """
    Get number of unmapped reads from an indexed BAM file.

    Args:
        bam_file_name (str): Name of indexed BAM file.

    Returns:
        int: number of unmapped reads in the BAM

    Note:
        BAM must be indexed for lookup using samtools.
    """

    index_output = tk_subproc.check_output(["samtools", "idxstats", bam_file_name])
    return int(index_output.strip().split(b"\n")[-1].split()[-1])


def kwargs_to_command_line_options(
    reserved_arguments: Union[List[str], Set[str]] = None,
    sep: str = " ",
    long_prefix: str = "--",
    short_prefix: str = "-",
    replace_chars: Optional[Dict[str, str]] = None,
    **kwargs,
) -> List[str]:
    """
    Convert arguments provided by user to a string usable in command line arguments.

    Args:
        reserved_arguments (set or list of str): set of arguments that this function prohibited for user use
        sep (str): separator between option/value pairs (`'='` for `--jobmode=sge`).
                   WARNING: switch args (`--abc`), which take no value, break if sep is not `' '`
        long_prefix (str): prefix for options with more than one character (`"--"` for `--quiet`, for example)
        short_prefix (str): prefix for options with one character (`"-"` for `-q`, for example)
        replace_chars (dict): map of characters to replace in specified variable names
                            (if `--align-reads` is command-line option, specify `align_reads` with `replace_chars` -> `{'_':'-'}`
        **kwargs (dict): `**kwargs` arguments/values to format string for

    Returns:
        list of str: Formatted arguments for use on a command line option.

    Raises:
        ValueError: if user requested argument conflicts with one of the
                    specified reserved arguments.
    """

    arguments = []
    reserved_arguments = set(reserved_arguments) if reserved_arguments else []
    replace_chars = replace_chars.items() if isinstance(replace_chars, dict) else []

    for key, value in kwargs.items():
        normalized_key = key.strip("-")

        # Replace characters for formatting
        for char, new_char in replace_chars:
            normalized_key = normalized_key.replace(char, new_char)

        # Validate user inputs to make sure no blatant conflicts
        if normalized_key in reserved_arguments:
            raise ValueError(
                "Specified option conflicts with reserved argument: %s. \
                             Reserved arguments are: %s"
                % (normalized_key, ",".join(reserved_arguments))
            )

        # Correctly prefix arguments
        if len(key) > 1:
            prefix = long_prefix
        else:
            prefix = short_prefix

        argument = "%s%s" % (prefix, normalized_key)
        option_value = value or ""
        arguments.append("%s%s%s" % (argument, sep, option_value))

    return arguments


def load_barcode_csv(barcode_csv: Union[AnyStr, os.PathLike]) -> Dict[bytes, List[bytes]]:
    """ Load a csv file of (genome,barcode) """

    bcs_per_genome = collections.defaultdict(
        list
    )  # type: collections.defaultdict[bytes, List[bytes]]
    with open(barcode_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                raise ValueError("Bad barcode file: %s" % barcode_csv)
            (genome, barcode) = row
            bcs_per_genome[genome.encode()].append(barcode.encode())
    return bcs_per_genome


def get_cell_associated_barcode_set(
    barcode_csv_filename: Union[AnyStr, os.PathLike], genome: Optional[bytes] = None
) -> Set[bytes]:
    """Get set of cell-associated barcode strings.

    Args:
      genome (bytes): Only get cell-assoc barcodes for this genome. If None, disregard genome.

    Returns:
      set of bytes: Cell-associated barcode strings (seq and gem-group).
    """
    cell_bcs_per_genome = load_barcode_csv(barcode_csv_filename)  # type: Dict[bytes, List[bytes]]
    cell_bcs = set()  # type: Set[bytes]
    for g, bcs in cell_bcs_per_genome.items():
        if genome is None or g == ensure_binary(genome):
            cell_bcs |= set(bcs)
    return cell_bcs


def string_is_ascii(input_string: AnyStr) -> bool:
    """Input strings are often stored as ascii in numpy arrays, and we need
    to check that this conversion works."""
    try:
        np.array(ensure_text(input_string, "utf-8"), dtype=bytes)
        return True
    except:
        return False


def chunk_reference(
    input_bam: AlignmentFile, nchunks: int, include_unmapped: bool
) -> List[List[Tuple[str, int, int]]]:
    """Chunk up a reference into nchunks roughly equally sized chunks.

    Args:
        input_bam (pysam.AlignmentFile): Source for reference contig names and lengths.
        nchunks (int): The number of chunks to create.
        include_unmapped (bool): If `True` then one of the chunks consists
                                 of all the unmapped reads as specified by
                                 `[("*", None, None)]`.

    Returns:
        list: loci, where each loci is a list of contiguous regions
              `(contig, start, end)`.
    """
    # one chunk is for unmapped reads
    nchunks -= int(include_unmapped)
    assert nchunks >= 1

    chroms = input_bam.references
    chrom_lengths = input_bam.lengths
    genome_size = sum(chrom_lengths)
    chunk_size = int(np.ceil(float(genome_size) / nchunks))
    process = [(chrom, 0, end) for chrom, end in zip(chroms, chrom_lengths)]

    chunks = []  # type: List[List[Tuple[str, int, int]]]
    if include_unmapped:
        chunks = [[("*", 0, 0)]]
    chunks.append([])

    current_chunk_size = 0
    while len(process):
        piece = process.pop()
        piece_size = piece[2] - piece[1]
        if current_chunk_size + piece_size < chunk_size:
            chunks[-1].append(piece)
            current_chunk_size += piece_size
        else:
            piece_left_over = current_chunk_size + piece_size - chunk_size
            new_piece = (piece[0], piece[1], piece[2] - piece_left_over)
            chunks[-1].append(new_piece)
            chunks.append([])
            current_chunk_size = 0
            if piece_left_over > 0:
                process.append((piece[0], piece[2] - piece_left_over, piece[2]))
    return [c for c in chunks if len(c)]
