# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
from __future__ import division, absolute_import, print_function
import enum
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy import interpolate
import scipy.stats as sp_stats
from six.moves import xrange
from six import iteritems, iterkeys, PY3

import martian

import bmkcc.cellranger.rna.library as rna_library
import bmkcc.cellranger.cell_calling as cr_cell
import bmkcc.cellranger.feature.antibody.analysis as ab_utils

from bmkcc.cellranger.constants import DEFAULT_RECOVERED_CELLS_PER_GEM_GROUP
from bmkcc.cellranger.metrics import BarcodeFilterResults
from bmkcc.tenkit.stats import robust_divide

if PY3:
    from typing import Dict, Iterable, Union  # pylint: disable=unused-import,import-error

ORDMAG_NUM_BOOTSTRAP_SAMPLES = 100
ORDMAG_RECOVERED_CELLS_QUANTILE = 0.99
NP_SORT_KIND = "stable"


class FilterMethod(enum.Enum):
    # Caller-provided list of barcodes
    MANUAL = 0
    # Take the top N barcodes by count
    TOP_N_BARCODES = 1
    # Take barcodes within an order of magnitude of the max by count
    ORDMAG = 2
    # The above (ORDMAG), then find barcodes that differ from the ambient profile
    ORDMAG_NONAMBIENT = 3
    # The above (ORDMAG), then find barcodes above the steepest gradient in the log-log rank plot
    GRADIENT = 4
    # Apply (GRADIENT) to total counts, then keep barcodes with target gene counts above minimum
    TARGETED = 5


def get_filter_method_name(fm):
    if fm == FilterMethod.MANUAL:
        return "manual"
    elif fm == FilterMethod.TOP_N_BARCODES:
        return "topn"
    elif fm == FilterMethod.ORDMAG:
        return "ordmag"
    elif fm == FilterMethod.ORDMAG_NONAMBIENT:
        return "ordmag_nonambient"
    elif fm == FilterMethod.GRADIENT:
        return "gradient"
    elif fm == FilterMethod.TARGETED:
        return "targeted"
    else:
        raise ValueError("Unsupported filter method value %d" % fm)


def get_filter_method_from_string(name):
    if name == "manual":
        return FilterMethod.MANUAL
    elif name == "topn":
        return FilterMethod.TOP_N_BARCODES
    elif name == "ordmag":
        return FilterMethod.ORDMAG
    elif name == "ordmag_nonambient":
        return FilterMethod.ORDMAG_NONAMBIENT
    elif name == "gradient":
        return FilterMethod.GRADIENT
    elif name == "targeted":
        return FilterMethod.TARGETED
    else:
        raise ValueError("Unknown filter method value %d" % name)


def validate_cell_calling_args(
    recovered_cells, force_cells, cell_barcodes, method_name, feature_types
):
    def throw_err(msg):
        raise ValueError(msg)

    # this will throw a ValueError if it's not a recognized error
    method = get_filter_method_from_string(method_name)

    if method == FilterMethod.ORDMAG or method == FilterMethod.ORDMAG_NONAMBIENT:
        pass

    elif method == FilterMethod.MANUAL:
        if cell_barcodes is None:
            throw_err("'cell_barcodes' must be specified when method is '%s'" % method_name)

    elif method == FilterMethod.TOP_N_BARCODES:
        if force_cells is None:
            throw_err("'force_cells' must be specified when method is '%s'" % method_name)


###############################################################################
def remove_bcs_with_high_umi_corrected_reads(correction_data, matrix):
    """Given a CountMatrix and and csv file containing information about umi corrected reads, detect
    1) all barcodes with unusually high fraction of corrected reads, and
    2) all barcodes with unusually high antibody UMI counts (protein aggregates),
    remove both from the CoutMatrix
    """

    augmented_table = ab_utils.augment_correction_table(
        correction_data, rna_library.ANTIBODY_LIBRARY_TYPE
    )
    # First, detect highly corrected barcodes
    highly_corrected_bcs = ab_utils.detect_highly_corrected_bcs(augmented_table)
    # Next, detect barcodes with high antibody UMI counts
    aggregate_bcs = ab_utils.detect_aggregate_barcodes(matrix)
    # Combine both sets and report a df of aggregate barcodes
    bcs_to_remove = highly_corrected_bcs + aggregate_bcs
    removed_bcs_df = ab_utils.subselect_augmented_table(bcs_to_remove, augmented_table)

    bcs_to_remove = set(matrix.bc_to_int(bc) for bc in bcs_to_remove)
    # make sure filtered_bcs is in deterministic order or any later bootstrap sampling will not be deterministic
    filtered_bcs = [i for i in xrange(matrix.bcs_dim) if i not in bcs_to_remove]
    cleaned_matrix = matrix.select_barcodes(filtered_bcs)

    ### report how many aggregates were found, and the fraction of reads those accounted for
    metrics_to_report = {}
    report_prefix = rna_library.get_library_type_metric_prefix(rna_library.ANTIBODY_LIBRARY_TYPE)
    metrics_to_report[report_prefix + "number_aggregate_GEMs"] = len(bcs_to_remove)
    frac_reads_removed = removed_bcs_df["Fraction Total Reads"].sum()
    metrics_to_report[report_prefix + "reads_lost_to_aggregate_GEMs"] = frac_reads_removed

    return cleaned_matrix, metrics_to_report, removed_bcs_df


################################################################################
def call_initial_cells(
    matrix,
    genomes,
    unique_gem_groups,
    method,
    recovered_cells,
    cell_barcodes,
    force_cells,
    feature_types,
    target_features=None,
):
    # make sure we have a CountMatrixView
    matrix = matrix.view()

    # (gem_group, genome) => dict
    filtered_metrics_groups = OrderedDict()
    # (gem_group, genome) => list of barcode strings
    filtered_bcs_groups = OrderedDict()

    if recovered_cells:
        gg_recovered_cells = recovered_cells // len(unique_gem_groups)
    else:
        gg_recovered_cells = None

    if force_cells:
        gg_force_cells = force_cells // len(unique_gem_groups)
    else:
        gg_force_cells = None
    for genome in genomes:
        # All sub-selection of the matrix should happen here & be driven by genome & feature_types
        genome_matrix = matrix.select_features_by_genome_and_types(genome, feature_types)

        # Make initial cell calls for each gem group individually
        for gem_group in unique_gem_groups:

            gg_matrix = genome_matrix.select_barcodes_by_gem_group(gem_group)
            gg_filtered_metrics, gg_filtered_bcs = _call_cells_by_gem_group(
                gg_matrix,
                method,
                gg_recovered_cells,
                gg_force_cells,
                cell_barcodes,
                target_features,
            )
            filtered_metrics_groups[(gem_group, genome)] = gg_filtered_metrics
            filtered_bcs_groups[(gem_group, genome)] = gg_filtered_bcs

    return filtered_metrics_groups, filtered_bcs_groups


def _call_cells_by_gem_group(
    gg_matrix, method, gg_recovered_cells, gg_force_cells, cell_barcodes, target_features,
):

    # counts per barcode
    gg_bc_counts = gg_matrix.get_counts_per_bc()

    if method == FilterMethod.ORDMAG or method == FilterMethod.ORDMAG_NONAMBIENT:

        gg_filtered_indices, gg_filtered_metrics, msg = filter_cellular_barcodes_ordmag(
            gg_bc_counts, gg_recovered_cells
        )
        gg_filtered_bcs = gg_matrix.ints_to_bcs(gg_filtered_indices)

    elif method == FilterMethod.MANUAL:
        bcs_in_matrix = set(gg_matrix.matrix.bcs)
        with (open(cell_barcodes)) as f:
            gg_filtered_bcs = [bc.encode() for bc in json.load(f)]
        gg_filtered_bcs = [bc for bc in gg_filtered_bcs if bc in bcs_in_matrix]
        gg_filtered_metrics = BarcodeFilterResults.init_with_constant_call(len(gg_filtered_bcs))
        msg = None

    elif method == FilterMethod.TOP_N_BARCODES:
        gg_filtered_indices, gg_filtered_metrics, msg = filter_cellular_barcodes_fixed_cutoff(
            gg_bc_counts, gg_force_cells
        )
        gg_filtered_bcs = gg_matrix.ints_to_bcs(gg_filtered_indices)

    elif method == FilterMethod.GRADIENT:
        gg_filtered_indices, gg_filtered_metrics, msg = filter_cellular_barcodes_gradient(
            gg_bc_counts, gg_recovered_cells
        )
        gg_filtered_bcs = gg_matrix.ints_to_bcs(gg_filtered_indices)

    elif method == FilterMethod.TARGETED:
        gg_bc_counts_targeted = gg_matrix.select_features(target_features).get_counts_per_bc()
        gg_filtered_indices, gg_filtered_metrics, msg = filter_cellular_barcodes_targeted(
            gg_bc_counts, gg_recovered_cells, gg_bc_counts_targeted
        )
        gg_filtered_bcs = gg_matrix.ints_to_bcs(gg_filtered_indices)

    else:
        martian.exit("Unsupported BC filtering method: %s" % method)
        raise SystemExit()

    if msg is not None:
        martian.log_info(msg)

    return gg_filtered_metrics, gg_filtered_bcs


def call_additional_cells(
    matrix, unique_gem_groups, genomes, filtered_bcs_groups, feature_types, chemistry_description
):
    # Track these for recordkeeping
    eval_bcs_arrays = []
    umis_per_bc_arrays = []
    loglk_arrays = []
    pvalue_arrays = []
    pvalue_adj_arrays = []
    nonambient_arrays = []
    genome_call_arrays = []

    matrix = matrix.select_features_by_types(feature_types)

    # Do it by gem group, but agnostic to genome
    for gg in unique_gem_groups:
        gg_matrix = matrix.select_barcodes_by_gem_group(gg)

        # Take union of initial cell calls across genomes
        gg_bcs = set()
        for group, bcs in iteritems(filtered_bcs_groups):
            if group[0] == gg:
                gg_bcs.update(bcs)
        gg_bcs = list(sorted(gg_bcs))

        result = cr_cell.find_nonambient_barcodes(gg_matrix, gg_bcs, chemistry_description)
        if result is None:
            print("Failed at attempt to call non-ambient barcodes in GEM well %s" % gg)
            continue

        # Assign a genome to the cell calls by argmax genome counts
        genome_counts = []
        for genome in genomes:
            genome_counts.append(
                gg_matrix.view()
                .select_features_by_genome(genome)
                .select_barcodes(result.eval_bcs)
                .get_counts_per_bc()
            )
        genome_counts = np.column_stack(genome_counts)
        genome_calls = np.array(genomes)[np.argmax(genome_counts, axis=1)]

        umis_per_bc = gg_matrix.get_counts_per_bc()

        eval_bcs_arrays.append(np.array(gg_matrix.bcs)[result.eval_bcs])
        umis_per_bc_arrays.append(umis_per_bc[result.eval_bcs])
        loglk_arrays.append(result.log_likelihood)
        pvalue_arrays.append(result.pvalues)
        pvalue_adj_arrays.append(result.pvalues_adj)
        nonambient_arrays.append(result.is_nonambient)
        genome_call_arrays.append(genome_calls)

        # Update the lists of cell-associated barcodes
        for genome in genomes:
            eval_bc_strs = np.array(gg_matrix.bcs)[result.eval_bcs]
            filtered_bcs_groups[(gg, genome)].extend(
                eval_bc_strs[(genome_calls == genome) & (result.is_nonambient)]
            )

    if len(eval_bcs_arrays) > 0:
        nonambient_summary = pd.DataFrame(
            OrderedDict(
                [
                    ("barcode", np.concatenate(eval_bcs_arrays)),
                    ("umis", np.concatenate(umis_per_bc_arrays)),
                    ("ambient_loglk", np.concatenate(loglk_arrays)),
                    ("pvalue", np.concatenate(pvalue_arrays)),
                    ("pvalue_adj", np.concatenate(pvalue_adj_arrays)),
                    ("nonambient", np.concatenate(nonambient_arrays)),
                    ("genome", np.concatenate(genome_call_arrays)),
                ]
            )
        )
    else:
        nonambient_summary = pd.DataFrame()

    return filtered_bcs_groups, nonambient_summary


################################################################################
def merge_filtered_metrics(filtered_metrics):
    # type: (Iterable[BarcodeFilterResults]) -> Dict[str, Union[float, int]]
    """ Merge all the barcode filter results and return them as a dictionary """
    result = BarcodeFilterResults(0)
    dresult = {}  # type: Dict[str, Union[float, int]]
    for i, fm in enumerate(filtered_metrics):
        dresult.update(fm.to_dict_with_prefix(i + 1))
        # Compute metrics over all gem groups
        result.filtered_bcs += fm.filtered_bcs
        result.filtered_bcs_lb += fm.filtered_bcs_lb
        result.filtered_bcs_ub += fm.filtered_bcs_ub
        result.filtered_bcs_var += fm.filtered_bcs_var

    # Estimate CV based on sum of variances and means
    result.filtered_bcs_cv = robust_divide(np.sqrt(result.filtered_bcs_var), result.filtered_bcs)

    dresult.update(result.__dict__)
    return dresult


def combine_initial_metrics(genomes, filtered_metrics_groups, genome_filtered_bcs, method, summary):
    # Combine initial-cell-calling metrics
    for genome in genomes:
        # Merge metrics over all gem groups for this genome
        txome_metrics = (v for k, v in iteritems(filtered_metrics_groups) if k[1] == genome)
        txome_summary = merge_filtered_metrics(txome_metrics)

        prefix = genome + "_" if genome else ""
        # Append method name to metrics
        summary.update(
            {
                ("%s%s_%s" % (prefix, key, get_filter_method_name(method))): txome_summary[key]
                for key in iterkeys(txome_summary)
            }
        )

        summary["%sfiltered_bcs" % prefix] = len(genome_filtered_bcs.get(genome, {}))

        # NOTE: This metric only applies to the initial cell calls
        summary["%sfiltered_bcs_cv" % prefix] = txome_summary["filtered_bcs_cv"]

    return summary


def summarize_bootstrapped_top_n(top_n_boot, nonzero_counts):
    top_n_bcs_mean = np.mean(top_n_boot)
    top_n_bcs_var = np.var(top_n_boot)
    top_n_bcs_sd = np.sqrt(top_n_bcs_var)
    result = BarcodeFilterResults()
    result.filtered_bcs_var = top_n_bcs_var
    result.filtered_bcs_cv = robust_divide(top_n_bcs_sd, top_n_bcs_mean)
    result.filtered_bcs_lb = np.round(sp_stats.norm.ppf(0.025, top_n_bcs_mean, top_n_bcs_sd), 0)
    result.filtered_bcs_ub = np.round(sp_stats.norm.ppf(0.975, top_n_bcs_mean, top_n_bcs_sd), 0)

    nbcs = int(np.round(top_n_bcs_mean))
    result.filtered_bcs = nbcs

    # make sure that if a barcode with count x is selected, we select all barcodes with count >= x
    # this is true for each bootstrap sample, but is not true when we take the mean

    if nbcs > 0:
        order = np.argsort(nonzero_counts, kind=NP_SORT_KIND)[::-1]
        sorted_counts = nonzero_counts[order]

        cutoff = sorted_counts[nbcs - 1]
        index = nbcs - 1
        if cutoff > 0:
            while (index + 1) < len(sorted_counts) and sorted_counts[index] == cutoff:
                index += 1
                # if we end up grabbing too many barcodes, revert to initial estimate
                if (index + 1 - nbcs) > 0.20 * nbcs:
                    return result
        result.filtered_bcs = index + 1

    return result


def find_within_ordmag(x, baseline_idx):
    x_ascending = np.sort(x)
    baseline = x_ascending[-baseline_idx]
    cutoff = int(max(1, np.round(0.1 * baseline)))
    # Return the index corresponding to the cutoff in descending order
    return len(x) - np.searchsorted(x_ascending, cutoff)


def filter_cellular_barcodes_ordmag(bc_counts, recovered_cells):
    """Simply take all barcodes that are within an order of magnitude of a top barcode
    that likely represents a cell
    """
    rs = np.random.RandomState(0)

    if recovered_cells is None:
        recovered_cells = DEFAULT_RECOVERED_CELLS_PER_GEM_GROUP

    metrics = BarcodeFilterResults(0)

    nonzero_bc_counts = bc_counts[bc_counts > 0]
    if len(nonzero_bc_counts) == 0:
        msg = "WARNING: All barcodes do not have enough reads for ordmag, allowing no bcs through"
        return [], metrics, msg

    baseline_bc_idx = int(np.round(float(recovered_cells) * (1 - ORDMAG_RECOVERED_CELLS_QUANTILE)))
    baseline_bc_idx = min(baseline_bc_idx, len(nonzero_bc_counts) - 1)

    # Bootstrap sampling; run algo with many random samples of the data
    top_n_boot = np.array(
        [
            find_within_ordmag(
                rs.choice(nonzero_bc_counts, len(nonzero_bc_counts)), baseline_bc_idx
            )
            for _ in xrange(ORDMAG_NUM_BOOTSTRAP_SAMPLES)
        ]
    )

    metrics.update(summarize_bootstrapped_top_n(top_n_boot, nonzero_bc_counts))

    # Get the filtered barcodes
    top_n = metrics.filtered_bcs
    top_bc_idx = np.sort(np.argsort(bc_counts, kind=NP_SORT_KIND)[::-1][0:top_n])
    assert top_n <= len(nonzero_bc_counts), "Invalid selection of 0-count barcodes!"
    return top_bc_idx, metrics, None


def filter_cellular_barcodes_fixed_cutoff(bc_counts, cutoff):
    nonzero_bcs = len(bc_counts[bc_counts > 0])
    top_n = min(cutoff, nonzero_bcs)
    top_bc_idx = np.sort(np.argsort(bc_counts, kind=NP_SORT_KIND)[::-1][0:top_n])
    metrics = BarcodeFilterResults.init_with_constant_call(top_n)
    return top_bc_idx, metrics, None


def filter_cellular_barcodes_targeted(
    bc_counts,
    recovered_cells,
    bc_counts_targeted,
    max_num_additional_cells=cr_cell.N_CANDIDATE_BARCODES,
    min_umis_additional_cells=cr_cell.TARGETED_CC_MIN_UMIS_ADDITIONAL_CELLS,
    min_targeted_umis=cr_cell.TARGETED_CC_MIN_UMIS_FROM_TARGET_GENES,
):
    """First, call cells using the gradient method (filter_cellular_barcodes_gradient) on the total bc_counts from all genes.
    Then retain all cell-associated barcodes which pass the min_targeted_umis threshold of UMI counts from target genes.
    """
    grad_cells_idx, _, msg = filter_cellular_barcodes_gradient(
        bc_counts, recovered_cells, max_num_additional_cells, min_umis_additional_cells
    )
    if msg is not None:
        martian.log_info(msg)

    targ_cells_idx = [idx for idx in grad_cells_idx if bc_counts_targeted[idx] >= min_targeted_umis]
    num_targ_cells = len(targ_cells_idx)
    metrics = BarcodeFilterResults.init_with_constant_call(num_targ_cells)
    return targ_cells_idx, metrics, None


def filter_cellular_barcodes_gradient(
    bc_counts,
    recovered_cells,
    max_num_additional_cells=cr_cell.N_CANDIDATE_BARCODES,
    min_umis_additional_cells=cr_cell.TARGETED_CC_MIN_UMIS_ADDITIONAL_CELLS,
):
    """Take all barcodes with counts above the value corresponding to the gradient of steepest decline
    in the log-transformed barcode rank plot [log(counts) vs log(rank) sorted by descending counts].
    This minimum gradient is computed over the allowed x-range based on ordmag cutoff and
    the allowable number of additional cells to consider. Prior to computing this value, the
    barcode rank plot is fit to a smoothing spline curve, and the interpolated first derivative of this
    curve is used to identify the point associated with maximum descent.
    """
    if recovered_cells is None:
        recovered_cells = DEFAULT_RECOVERED_CELLS_PER_GEM_GROUP

    metrics = BarcodeFilterResults(0)

    nonzero_bc_counts = bc_counts[bc_counts > 0]
    nonzero_bc_counts = np.array(sorted(nonzero_bc_counts)[::-1])  # sort in descending order
    if len(nonzero_bc_counts) == 0:
        msg = "WARNING: All barcodes do not have enough reads for gradient, allowing no bcs through"
        return [], metrics, msg

    baseline_bc_idx = int(np.round(float(recovered_cells) * (1 - ORDMAG_RECOVERED_CELLS_QUANTILE)))
    baseline_bc_idx = min(baseline_bc_idx, len(nonzero_bc_counts) - 1)
    baseline_count_threshold = nonzero_bc_counts[baseline_bc_idx]

    lower_bc_idx = np.sum(nonzero_bc_counts >= baseline_count_threshold / 10.0) - 1
    lower_bc_idx = min(lower_bc_idx, len(nonzero_bc_counts) - 1)

    upper_bc_idx = min(
        lower_bc_idx + max_num_additional_cells,
        np.sum(nonzero_bc_counts >= min_umis_additional_cells),
    )
    upper_bc_idx = max(upper_bc_idx, lower_bc_idx)
    upper_bc_idx = min(upper_bc_idx, len(nonzero_bc_counts) - 1)

    uniq_counts = sorted(set(nonzero_bc_counts))[
        ::-1
    ]  # collapse unique values and sort in descending order
    log_y_values = [np.log10(a) for a in uniq_counts]
    x_values = [np.sum(nonzero_bc_counts >= a) for a in uniq_counts]
    log_x_values = [np.log10(x) for x in x_values]
    log_x_values.append(
        np.log10(1 + sum(nonzero_bc_counts))
    )  # append end-point values to handle small input arrays
    log_y_values.append(0.0)

    # Fit spline to barcode rank plot curve, then interpolate its first derivative over log_x_values
    spline_degree = min(3, len(log_y_values) - 1)  # adjust degree parameter for small input arrays
    fit_spline = interpolate.UnivariateSpline(
        x=log_x_values, y=log_y_values, k=spline_degree, s=0, check_finite=True
    )
    # For large input arrays, reduce the number of knots for progressive smoothing
    if len(log_x_values) > 50:
        num_spline_knots = get_spline_num_knots(len(log_x_values))
        orig_knots = fit_spline.get_knots()
        if num_spline_knots < len(orig_knots):
            knot_values = [
                orig_knots[i]
                for i in np.linspace(1, len(orig_knots) - 2, num_spline_knots - 2, dtype=int)
            ]
            fit_spline = interpolate.LSQUnivariateSpline(
                x=log_x_values, y=log_y_values, t=knot_values, k=spline_degree, check_finite=True
            )

    # Find minimum gradient value for barcodes within the allowed x-range
    gradients = fit_spline(log_x_values[0:-1], 1)
    gradients = np.where(
        [x >= lower_bc_idx and x <= upper_bc_idx for x in x_values],
        gradients,
        np.repeat(0, len(gradients)),
    )
    gradient_count_cutoff = np.round(10 ** log_y_values[np.argmin(gradients)], 0)
    gradient_num_cells = max(np.sum(nonzero_bc_counts > gradient_count_cutoff), lower_bc_idx + 1)

    # Get filtered barcodes corresponding to gradient-based cutoff
    top_n = min(gradient_num_cells, len(nonzero_bc_counts))
    top_bc_idx = np.sort(np.argsort(bc_counts, kind=NP_SORT_KIND)[::-1][0:top_n])
    metrics = BarcodeFilterResults.init_with_constant_call(top_n)
    return top_bc_idx, metrics, None


def get_spline_num_knots(n):
    """Heuristic function to estimate the number of knots to be used for spline interpolation
    as a function of the number of unique input data points n.
    """
    if n < 50:
        return int(n)
    a1 = np.log2(50)
    a2 = np.log2(100)
    a3 = np.log2(140)
    a4 = np.log2(200)
    if n < 200:
        return int(2 ** (a1 + (a2 - a1) * (n - 50) / 150))
    if n < 800:
        return int(2 ** (a2 + (a3 - a2) * (n - 200) / 600))
    if n < 3200:
        return int(2 ** (a3 + (a4 - a3) * (n - 800) / 2400))
    return int(200 + (n - 3200) ** (0.2))
