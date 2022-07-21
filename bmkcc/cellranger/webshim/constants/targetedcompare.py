#!/usr/bin/env python
#
# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
#

# SINGLE CELL
def tc_metrics(is_spatial):
    if is_spatial:
        CELL_OR_TISSUE = "tissue"
        CELL_OR_SPOT = "Spot"
        TAS_OR_CELL = "tissue-associated spot"
        TAS_OR_CCP = "tissue-covered spot"
    else:
        CELL_OR_TISSUE = "cell"
        CELL_OR_SPOT = "Cell"
        TAS_OR_CELL = "cell"
        TAS_OR_CCP = "cell-containing partitions"

    NUM_GENES_ENRICHED_METRIC = {
        "name": "num_targeted_genes_enriched",
        "display_name": "Number of Targeted Genes Enriched Relative to Parent",
        "description": "Number of Targeted Genes Enriched. Only genes that have at least 1 "
        + CELL_OR_TISSUE
        + "-associated UMI in the parent sample are considered. Genes are classified with a two-class Gaussian mixture model according to their read enrichment relative to the parent sample. Genes in the group with the higher read enrichment are classified as enriched.",
        "format": "integer",
    }

    CELL_RELATIVE_TARGETED_DEPTH_METRIC = {
        "name": "cell_targeted_depth_factor",
        "display_name": "Mean Targeted Reads per " + CELL_OR_SPOT + " Relative to Parent",
        "description": "Mean ratio of targeted reads per  "
        + TAS_OR_CELL
        + " in the targeted sample relative to parent sample for all barcodes called as a "
        + TAS_OR_CELL
        + " in both samples.",
        "format": "%.2f",
    }

    TARGETED_ENRICHMENT_METRICS = [
        NUM_GENES_ENRICHED_METRIC,
        CELL_RELATIVE_TARGETED_DEPTH_METRIC,
    ]

    TOTAL_READ_PAIRS_TARGETED_METRIC = {
        "name": "total_read_pairs_in_targeted_sample",
        "display_name": "Number of Reads in Targeted Sample",
        "description": "Total number of read pairs in the targeted sample that were assigned to gene expression libraries in demultiplexing.",
        "format": "integer",
    }

    TOTAL_READ_PAIRS_PARENT_METRIC = {
        "name": "total_read_pairs_in_parent_sample",
        "display_name": "Number of Reads in Parent Sample",
        "description": "Total number of read pairs in the parent sample that were assigned to gene expression libraries in demultiplexing.",
        "format": "integer",
    }

    FRAC_ON_TARGET_TRANSCRIPTOME_TARGETED_METRIC = {
        "name": "fraction_reads_on_target_in_targeted_sample",
        "display_name": "Reads Confidently Mapped to the Targeted Transcriptome in Targeted Sample",
        "description": "Fraction of reads that mapped to a unique and targeted gene in the transcriptome. The read must be consistent with annotated splice junctions. These reads are considered for UMI counting.",
        "format": "percent",
    }

    FRAC_ON_TARGET_TRANSCRIPTOME_PARENT_METRIC = {
        "name": "fraction_reads_on_target_in_parent_sample",
        "display_name": "Reads Confidently Mapped to the Targeted Transcriptome in Parent Sample",
        "description": "Fraction of reads that mapped to a unique and targeted gene in the transcriptome. The read must be consistent with annotated splice junctions. These reads are considered for UMI counting.",
        "format": "percent",
    }

    NUM_CELLS_TARGETED_METRIC = {
        "name": "num_cells_in_targeted_sample",
        "display_name": "Number of  " + TAS_OR_CELL.title() + "s Called in Targeted Sample",
        "description": "Number of barcodes called as  " + TAS_OR_CELL + "s in the Targeted Sample.",
        "format": "integer",
    }

    NUM_CELLS_PARENT_METRIC = {
        "name": "num_cells_in_parent_sample",
        "display_name": "Number of " + TAS_OR_CELL.title() + "s Called in Parent Sample",
        "description": "Number of barcodes called as " + TAS_OR_CELL + "s in the Parent Sample.",
        "format": "integer",
    }

    MEAN_RPC_TARGETED_METRIC = {
        "name": "mean_reads_per_cell_in_targeted_sample",
        "display_name": "Mean Reads per " + CELL_OR_SPOT + " in Targeted Sample",
        "description": "The total number of reads divided by the number of barcodes associated with "
        + TAS_OR_CCP
        + "s in the targeted sample.",
        "format": "integer",
    }

    MEAN_RPC_PARENT_METRIC = {
        "name": "mean_reads_per_cell_in_parent_sample",
        "display_name": "Mean Reads per " + CELL_OR_SPOT + " in Parent Sample",
        "description": "The total number of reads divided by the number of barcodes associated with "
        + TAS_OR_CCP
        + " in the parent sample.",
        "format": "integer",
    }

    MEAN_PRPC_TARGETED_METRIC = {
        "name": "mean_targeted_reads_per_cell_in_targeted_sample",
        "display_name": "Mean Targeted Reads per " + CELL_OR_SPOT + " in Targeted Sample",
        "description": "The total number of targeted reads divided by the number of barcodes associated with "
        + TAS_OR_CCP
        + "s in the targeted sample.",
        "format": "integer",
    }

    MEAN_PRPC_PARENT_METRIC = {
        "name": "mean_targeted_reads_per_cell_in_parent_sample",
        "display_name": "Mean Targeted Reads per " + CELL_OR_SPOT + " in Parent Sample",
        "description": "The total number of targeted reads divided by the number of barcodes associated with "
        + TAS_OR_CCP
        + "s in the parent sample.",
        "format": "integer",
    }

    MEDIAN_TARGETED_GENES_PER_CELL_TARGETED_METRIC = {
        "name": "median_targeted_genes_per_cell_in_targeted_sample",
        "display_name": "Median Targeted Genes per " + CELL_OR_SPOT + " in Targeted Sample",
        "description": "The median number of targeted genes detected per "
        + TAS_OR_CELL
        + " barcode in the targeted sample.",
        "format": "integer",
    }

    MEDIAN_TARGETED_GENES_PER_CELL_PARENT_METRIC = {
        "name": "median_targeted_genes_per_cell_in_parent_sample",
        "display_name": "Median Targeted Genes per " + CELL_OR_SPOT + " in Parent Sample",
        "description": "The median number of targeted genes detected per "
        + TAS_OR_CELL
        + "  barcode in the parent sample.",
        "format": "integer",
    }

    NUM_TARGETED_GENES_DETECTED_TARGETED_METRIC = {
        "name": "total_targeted_genes_detected_in_targeted_sample",
        "display_name": "Number of Targeted Genes Detected in Targeted Sample",
        "description": "Number of Targeted Genes Detected in Targeted Sample. A gene is considered detected if it has at least 1 "
        + CELL_OR_TISSUE
        + "-associated UMI.",
        "format": "integer",
    }

    NUM_TARGETED_GENES_DETECTED_PARENT_METRIC = {
        "name": "total_targeted_genes_detected_in_parent_sample",
        "display_name": "Number of Targeted Genes Detected in Parent Sample",
        "description": "Number of Targeted Genes Detected in Parent Sample. A gene is considered detected if it has at least 1 "
        + CELL_OR_TISSUE
        + "-associated UMI.",
        "format": "integer",
    }

    NUM_TARGETED_GENES_EXCLUSIVE_TARGETED_METRIC = {
        "name": "num_targeted_genes_detected_exclusive_in_targeted_sample",
        "display_name": "Number of Targeted Genes Detected Exclusively in Targeted Sample",
        "description": "Number of Targeted Genes Detected Exclusively in Targeted Sample. A gene is considered detected if it has at least 1 UMI count in "
        + CELL_OR_TISSUE
        + "-associated barcodes.",
        "format": "integer",
    }

    NUM_TARGETED_GENES_EXCLUSIVE_PARENT_METRIC = {
        "name": "num_targeted_genes_detected_exclusive_in_parent_sample",
        "display_name": "Number of Targeted Genes Detected Exclusively in Parent Sample",
        "description": "Number of Targeted Genes Detected Exclusively in Parent Sample. A gene is considered detected if it has at least 1 UMI count in "
        + CELL_OR_TISSUE
        + "-associated barcodes.",
        "format": "integer",
    }

    MEDIAN_TARGETED_UMIS_PER_CELL_TARGETED_METRIC = {
        "name": "median_targeted_umis_per_cell_in_targeted_sample",
        "display_name": "Median Targeted UMIs per " + CELL_OR_SPOT + " in Targeted Sample",
        "description": "The median number of UMI counts in targeted genes per "
        + CELL_OR_TISSUE
        + "-associated barcode in the targeted sample.",
        "format": "integer",
    }

    MEDIAN_TARGETED_UMIS_PER_CELL_PARENT_METRIC = {
        "name": "median_targeted_umis_per_cell_in_parent_sample",
        "display_name": "Median Targeted UMIs per " + CELL_OR_SPOT + " in Parent Sample",
        "description": "The median number of UMI counts in targeted genes per "
        + CELL_OR_TISSUE
        + "-associated barcode in the parent sample.",
        "format": "integer",
    }

    PAIRED_METRICS = [
        TOTAL_READ_PAIRS_TARGETED_METRIC,
        TOTAL_READ_PAIRS_PARENT_METRIC,
        FRAC_ON_TARGET_TRANSCRIPTOME_TARGETED_METRIC,
        FRAC_ON_TARGET_TRANSCRIPTOME_PARENT_METRIC,
        NUM_CELLS_TARGETED_METRIC,
        NUM_CELLS_PARENT_METRIC,
        MEAN_RPC_TARGETED_METRIC,
        MEAN_RPC_PARENT_METRIC,
        MEAN_PRPC_TARGETED_METRIC,
        MEAN_PRPC_PARENT_METRIC,
        MEDIAN_TARGETED_GENES_PER_CELL_TARGETED_METRIC,
        MEDIAN_TARGETED_GENES_PER_CELL_PARENT_METRIC,
        NUM_TARGETED_GENES_DETECTED_TARGETED_METRIC,
        NUM_TARGETED_GENES_DETECTED_PARENT_METRIC,
        NUM_TARGETED_GENES_EXCLUSIVE_TARGETED_METRIC,
        NUM_TARGETED_GENES_EXCLUSIVE_PARENT_METRIC,
        MEDIAN_TARGETED_UMIS_PER_CELL_TARGETED_METRIC,
        MEDIAN_TARGETED_UMIS_PER_CELL_PARENT_METRIC,
    ]

    NUM_CELLS_BOTH = {
        "name": "num_cell_calls_both",
        "display_name": "Number of " + CELL_OR_SPOT + "s Called in Parent and Targeted Samples",
        "description": "Number of barcodes called as "
        + TAS_OR_CELL
        + "s in both the parent and targeted samples.",
        "format": "integer",
    }

    NUM_CELLS_BOTH_TARGETED_ONLY = {
        "name": "num_cell_calls_targeted-only",
        "display_name": "Number of " + TAS_OR_CELL.title() + "s Called only in Targeted Sample",
        "description": "Number of barcodes called as "
        + TAS_OR_CELL
        + "s in the Targeted Sample and not the Parent Sample.",
        "format": "integer",
    }

    NUM_CELLS_PARENT_ONLY = {
        "name": "num_cell_calls_parent-only",
        "display_name": "Number of " + TAS_OR_CELL.title() + "s Called only in Parent Sample",
        "description": "Number of barcodes called as "
        + TAS_OR_CELL
        + "s in the Parent Sample and not the Targeted Sample.",
        "format": "integer",
    }

    MEAN_READ_ENRICHMENT = {
        "name": "mean_read_enrichment",
        "display_name": "Mean Read Enrichment across Targeted Genes",
        "description": "Mean Read Enrichment across Targeted Genes. Enrichments are only computed for genes with at least 1 "
        + CELL_OR_TISSUE
        + "-associated UMI in both the targeted and parent samples. Mean is the geometric mean of the enrichments. Samples are rescaled to the total number of reads in order to account for differences in sequencing depth.",
        "format": "%.3f",
    }

    PER_GENE_READ_CORR = {
        "name": "targeted_gene_read_rsquared",
        "display_name": "Per-Gene Read Counts R-squared",
        "description": "Pearson correlation coefficient (squared) of the number of reads confidently mapped to targeted genes (log10) in the targeted vs parent experiments.",
        "format": "%.3f",
    }

    MEAN_UMI_RATIO = {
        "name": "mean_frac_UMIs_recovered_per_gene",
        "display_name": "Mean Ratio of Targeted UMI Counts in Targeted Sample Relative to Parent",
        "description": "Mean per-gene ratio of UMIs observed in targeted sample relative to the parent sample.",
        "format": "%.3f",
    }

    PER_GENE_UMI_CORR = {
        "name": "targeted_gene_umi_rsquared",
        "display_name": "Per-Gene UMI Counts R-squared",
        "description": "Pearson correlation coefficient (squared) of the number of UMIs confidently mapped to targeted genes (log10) in the targeted vs parent experiments.",
        "format": "%.3f",
    }

    PLOTTED_METRICS = [
        NUM_CELLS_BOTH,
        NUM_CELLS_BOTH_TARGETED_ONLY,
        NUM_CELLS_PARENT_ONLY,
        MEAN_READ_ENRICHMENT,
        PER_GENE_READ_CORR,
        MEAN_UMI_RATIO,
        PER_GENE_UMI_CORR,
    ]

    METRICS = [
        {"name": "Targeted Enrichment", "metrics": TARGETED_ENRICHMENT_METRICS},
        {"name": "Paired Metrics", "metrics": PAIRED_METRICS},
        {"name": "Plotted Metrics", "metrics": PLOTTED_METRICS},
    ]

    return METRICS
    # ALARMS = []
