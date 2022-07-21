#!/usr/bin/env python
#
# Copyright (c) 2014 10X Genomics, Inc. All rights reserved.
#

"""Values used in many other modules."""

######################################################
# DO NOT add new items to this file.
#
# - If a constant is only used from a single module, put it in that module.
# - If a constant is only used in association with a particular module, put it
#   in that module.
# - If a constant is used in a bunch of places, create a module with a more
#   specifically descriptive name to put those constants.
######################################################

# What is considered a high confidence mapped read pair
HIGH_CONF_MAPQ = 60

# Distance to mate to ensure the
MIN_MATE_OFFSET_DUP_FILTER = 20

# Sequencing settings
ILLUMINA_QUAL_OFFSET = 33
DEFAULT_HIGH_MAPQ = 60

# Demultiplex settings
# DEMULTIPLEX_DEFAULT_SAMPLE_INDEX_LENGTH = 8
DEMULTIPLEX_ACCEPTED_SAMPLE_INDEX_LENGTH = [8, 10]
""" New defaults for dual-index """
DEMULTIPLEX_BARCODE_LENGTH = 14
DEMULTIPLEX_INVALID_SAMPLE_INDEX = "X"
MAX_INDICES_TO_DEMUX = 1500
""" Maximum number of sample indices allowed """
SAMPLE_INDEX_MAX_HAMMING_DISTANCE = 1
""" Maximum separation in HD between SIs of the same kind i7/i5 """

# TAG names
PROCESSED_BARCODE_TAG = "BX"
SAMPLE_INDEX_TAG = "BC"
SAMPLE_INDEX_QUAL_TAG = "QT"
TRIM_TAG = "TR"
TRIM_QUAL_TAG = "TQ"

# Parallelization settings
PARALLEL_LOCUS_SIZE = int(4e7)

#
# Settings for metrics computation
#

# Distance to consider reads far away for far chimeras
READ_MATE_FAR_DIST = 5000

# Preflight constants
BCL_PROCESSOR_FASTQ_MODE = "BCL_PROCESSOR"
ILMN_BCL2FASTQ_FASTQ_MODE = "ILMN_BCL2FASTQ"
