#
# Copyright (c) 2020 10X Genomics, Inc. All rights reserved.
#

from __future__ import absolute_import

import h5py
import numpy as np
import six

from bmkcc.cellranger.wrapped_tables import tables
import bmkcc.cellranger.h5_constants as h5_constants

if six.PY3:
    from html import unescape  # pylint: disable=no-name-in-module
    from typing import List, Optional, Union  # pylint: disable=unused-import
else:
    # Instance for unescaping, so we don't need to create one every time we decode.
    _HTML_PARSER = six.moves.html_parser.HTMLParser()

    def unescape(in_str):  # pylint: disable=function-redefined
        """Convert all named and numeric character references (e.g.
        `&gt;`, `&#62;`, `&#x3e;`) in the string `in_str` to the
        corresponding Unicode characters.
        """
        return _HTML_PARSER.unescape(in_str)


STR_DTYPE_CHAR = "S"
UNICODE_DTYPE_CHAR = "U"
STRING_DTYPE_CHARS = (STR_DTYPE_CHAR, UNICODE_DTYPE_CHAR)


def is_hdf5(filename, throw_exception=False):
    """
    A wrapper around h5py.is_hdf5, optionally can throw an exception.

    Args:
        filename: The name of the file to test
        throw_exception: Should we raise an error if not?

    Returns:
        bool as to whether the file is valid
    """
    valid = h5py.is_hdf5(filename)
    if not valid and throw_exception:
        raise IOError("File: {} is not a valid HDF5 file.".format(filename))
    return valid


def write_h5(filename, data, append=False):
    filemode = "w"
    if append:
        filemode = "a"
    with h5py.File(filename, filemode) as f:
        for key, value in six.iteritems(data):
            f[key] = value


def get_h5_filetype(filename):
    with tables.open_file(filename, mode="r") as f:
        try:
            filetype = six.ensure_str(f.get_node_attr("/", h5_constants.H5_FILETYPE_KEY))
        except AttributeError:
            filetype = None  # older files lack this key
    return filetype


def save_array_h5(filename, name, arr):
    """ Save an array to the root of an h5 file """
    with tables.open_file(filename, "w") as f:
        f.create_carray(f.root, name, obj=arr)


def load_array_h5(filename, name):
    """ Load an array from the root of an h5 file """
    with tables.open_file(filename, "r") as f:
        return getattr(f.root, name).read()


def create_hdf5_string_dataset(group, name, data, **kwargs):
    """Create a dataset of strings under an HDF5 (h5py) group.

    Strings are stored as fixed-length 7-bit ASCII with XML-encoding
    for characters outside of 7-bit ASCII. This is inspired by the
    choice made for the Loom spec:
    https://github.com/linnarsson-lab/loompy/blob/master/doc/format/index.rst

    Args:
        group (h5py.Node): Parent group.
        name (str): Dataset name.
        data (list of str): Data to store. Both None and [] are serialized to an empty dataset.
                            Both elements that are empty strings and elements that are None are
                            serialized to empty strings.
    """

    if data is None or hasattr(data, "__len__") and len(data) == 0:
        dtype = "%s1" % STR_DTYPE_CHAR
        group.create_dataset(name, dtype=dtype)
        return

    assert (isinstance(data, np.ndarray) and data.dtype.char == STR_DTYPE_CHAR) or (
        isinstance(data, list)
        and all(x is None or isinstance(x, (bytes, six.text_type)) for x in data)
    )

    # Convert Nones to empty strings and use XML encoding
    data = [encode_ascii_xml(x) if x else b"" for x in data]

    fixed_len = max(len(x) for x in data)

    # h5py doesn't support strings with zero-length-dtype
    if fixed_len == 0:
        fixed_len = 1
    dtype = "%s%d" % (STR_DTYPE_CHAR, fixed_len)

    group.create_dataset(name, data=data, dtype=dtype, **kwargs)


def decode_ascii_xml(x):
    # type: (Union[str, bytes]) -> str
    """
    Decode a string from 7-bit ASCII + XML into unicode.
    """
    if isinstance(x, six.text_type):
        return x
    elif isinstance(x, six.binary_type):
        return six.ensure_text(unescape(six.ensure_str(x)))
    else:
        raise ValueError("Expected string type, got type %s" % str(type(x)))


def decode_ascii_xml_array(data):
    """Decode an array-like container of strings from 7-bit ASCII + XML
    into unicode.
    """
    if isinstance(data, np.ndarray) and data.dtype.char == UNICODE_DTYPE_CHAR:
        return data

    unicode_data = [decode_ascii_xml(x) for x in data]

    fixed_len = max(len(s) for s in unicode_data)
    # 0 length string type is Ok
    dtype = "%s%d" % (UNICODE_DTYPE_CHAR, fixed_len)

    # note: python3 would require no.fromiter
    return np.array(unicode_data, dtype=dtype)


def encode_ascii_xml(x):
    """Encode a string as fixed-length 7-bit ASCII with XML-encoding
    for characters outside of 7-bit ASCII.

    Respect python2 and python3, either unicode or binary.
    """
    if isinstance(x, six.text_type):
        return x.encode("ascii", "xmlcharrefreplace")
    elif isinstance(x, six.binary_type):
        return x
    else:
        raise ValueError("Expected string type, got type %s" % str(type(x)))


def encode_ascii_xml_array(data):
    """Encode an array-like container of strings as fixed-length 7-bit ASCII
    with XML-encoding for characters outside of 7-bit ASCII.
    """
    if (
        isinstance(data, np.ndarray)
        and data.dtype.char == STR_DTYPE_CHAR
        and data.dtype.itemsize > 0
    ):
        return data

    convert = lambda s: encode_ascii_xml(s) if s is not None else b""
    ascii_data = [convert(x) for x in data]

    fixed_len = max(len(s) for s in ascii_data)
    fixed_len = max(1, fixed_len)
    dtype = "%s%d" % (STR_DTYPE_CHAR, fixed_len)

    # note: python3 would require np.fromiter
    return np.array(ascii_data, dtype=dtype)


# Global variables used in memoization below
_mlastConverted = None
_mlastValue = None  # type: Optional[str]


def read_hdf5_string_dataset(dataset, memoize=False):
    # type: (h5py.Dataset, bool) -> List[str]
    """Read a dataset of strings from HDF5 (h5py).

    Args:
        dataset (h5py.Dataset): Data to read.
        memoize (bool): Whether to use memoization to make string conversion more efficient.

    Returns:
        list[unicode]: Strings in the dataset.
    """

    # h5py doesn't support loading an empty dataset
    if dataset.shape is None:
        return []

    data = dataset[:]
    decode = decode_ascii_xml
    if memoize:

        def decoder(x):
            # type: (Union[str, bytes]) -> str
            # pylint: disable=global-statement
            global _mlastConverted
            global _mlastValue
            if x != _mlastConverted:
                _mlastValue = decode_ascii_xml(x)
                _mlastConverted = x
            assert _mlastValue is not None
            return _mlastValue

        decode = decoder
    return [decode(x) for x in data]


def set_hdf5_attr(dataset, name, value):
    """Set an attribute of an HDF5 dataset/group.

    Strings are stored as fixed-length 7-bit ASCII with XML-encoding
    for characters outside of 7-bit ASCII. This is inspired by the
    choice made for the Loom spec:
    https://github.com/linnarsson-lab/loompy/blob/master/doc/format/index.rst
    """

    name = encode_ascii_xml(name)

    if isinstance(value, (six.text_type, six.binary_type)):
        value = encode_ascii_xml(value)
    elif isinstance(value, np.ndarray) and value.dtype.char in STRING_DTYPE_CHARS:
        value = encode_ascii_xml_array(value)
    elif isinstance(value, (list, tuple)) and isinstance(
        value[0], (six.text_type, six.binary_type)
    ):
        value = encode_ascii_xml_array(value)

    dataset.attrs[name] = value


def combine_h5s_into_one(outfile, files_to_combine):
    # type: (str, list) -> None
    assert isinstance(files_to_combine, list)
    with tables.open_file(six.ensure_binary(outfile), "a") as out:
        for fname in files_to_combine:
            with tables.open_file(six.ensure_binary(fname), "r") as data:
                data.copy_children(data.root, out.root, recursive=True)
