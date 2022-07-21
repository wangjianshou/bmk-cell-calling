#!/usr/bin/env python
#
# Copyright (c) 2014 10X Genomics, Inc. All rights reserved.
#
# General utilities for manipulating nucleotide sequences
#

from __future__ import absolute_import

import six

NUCS = [b"A", b"C", b"G", b"T"]
NUCS_INVERSE = {b"A": 0, b"C": 1, b"G": 2, b"T": 3}

if six.PY2:
    from string import maketrans  # pylint: disable=no-name-in-module
else:
    from typing import AnyStr  # pylint: disable=unused-import

    maketrans = bytes.maketrans  # pylint: disable=no-member

DNA_CONVERT_TABLE = maketrans(  # pylint: disable=deprecated-string-function
    b"ACGTacgtRYMKBDHVrymkbdhv", b"TGCAtgcaYRKMVHDByrkmvhdb"
)
RNA_CONVERT_TABLE = maketrans(  # pylint: disable=deprecated-string-function
    b"ACGUacguRYMKBDHVrymkbdhv", b"UGCAugcaYRKMVHDByrkmvhdb"
)

IUPAC_NUC_MAP = {
    b"A": [b"A"],
    b"C": [b"C"],
    b"G": [b"G"],
    b"T": [b"T"],
    b"R": [b"A", b"G"],
    b"Y": [b"C", b"T"],
    b"M": [b"C", b"A"],
    b"K": [b"T", b"G"],
    b"W": [b"T", b"A"],
    b"S": [b"C", b"G"],
    b"B": [b"C", b"T", b"G"],
    b"D": [b"T", b"A", b"G"],
    b"H": [b"T", b"A", b"C"],
    b"V": [b"A", b"C", b"G"],
    b"N": [b"T", b"A", b"C", b"G"],
}


def get_rev_comp(seq):
    # type: (AnyStr) -> bytes
    """Reverse complement for DNA.  Included ambiguous nucleotides and retains case."""
    return six.ensure_binary(seq).translate(DNA_CONVERT_TABLE)[::-1]


def mask(seq, keep_start, keep_end):
    # type: (bytes, int, int) -> bytes
    """Mask the sequence leaving only [keep_start, keep_end) unmasked"""
    return b"N" * keep_start + seq[keep_start:keep_end] + b"N" * (len(seq) - keep_end)
