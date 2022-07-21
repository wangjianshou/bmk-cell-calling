#!/usr/bin/env python
#
# Copyright (c) 2014 10X Genomics, Inc. All rights reserved.
#

"""Methods for safely encoding values to json.

The json standard does not permit encoding NaN, but Python will still happily
do it, which can cause problems for other programs.  This module contains code
to fix those values up, as well as convert numpy objects to things which encode
properly in json.
"""

from __future__ import absolute_import

import json
import math
import numpy as np
import six

NAN_STRING = "NaN"
POS_INF_STRING = "inf"
NEG_INF_STRING = "-inf"
_NEG_INF = float("-Inf")
_POS_INF = float("+Inf")


def desanitize_value(x):
    """
    Converts back a special numeric value encoded by json_sanitize
    :param x: a string encoded value or float
    :return: a float or the original value
    """
    if x == NAN_STRING:
        return float("NaN")
    elif x == POS_INF_STRING:
        return _POS_INF
    elif x == NEG_INF_STRING:
        return _NEG_INF
    else:
        return x


def json_sanitize(data):
    """
    Yuck yuck yuck yuck yuck!
    The default JSON encoders do bad things if you try to encode
    NaN/Infinity/-Infinity as JSON.  This code takes a nested data structure
    composed of: atoms, dicts, or array-like iteratables and finds all of the
    not-really-a-number floats and converts them to an appropriate string for
    subsequent jsonification.
    """
    # This really doesn't make me happy. How many cases we we have to test?
    if isinstance(data, str):
        return data
    elif isinstance(data, bytes):
        return six.ensure_str(data)
    elif isinstance(data, (float, np.float64)):
        # Handle floats specially
        if math.isnan(data):
            return NAN_STRING
        if data == float("+Inf"):
            return POS_INF_STRING
        if data == float("-Inf"):
            return NEG_INF_STRING
        return data
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, dict):
        return {json_sanitize(k): json_sanitize(value) for k, value in data.items()}
    elif hasattr(data, "keys"):
        # Dictionary-like case
        return {json_sanitize(k): json_sanitize(data[k]) for k in data.keys()}
    elif hasattr(data, "__iter__"):
        # Anything else that looks like a list. N
        return [json_sanitize(item) for item in data]
    elif hasattr(data, "shape") and data.shape == ():
        # Numpy 0-d array
        return np.asscalar(data)
    else:
        return data


def safe_jsonify(data, pretty=False, **kwargs):
    """Dump an object to a string as json, after sanitizing it."""
    safe_data = json_sanitize(data)
    if pretty:
        return json.dumps(
            safe_data,
            indent=kwargs.pop("indent", 4),
            sort_keys=kwargs.pop("sort_keys", True),
            separators=kwargs.pop("separators", (",", ":")),
            **kwargs,
        )
    else:
        return json.dumps(safe_data, **kwargs)


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """This encoder will convert 1D np.ndarrays to lists.  For other numpy
    types, uses obj.item() to extract a python scalar."""

    def default(self, o):  # pylint: disable=method-hidden
        """Convert a 1D np.ndarray into a list, or a single-element array or
        matrix into its scalar value."""
        if isinstance(o, np.ndarray) and o.ndim == 1:
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.generic):
            return o.item()
        elif isinstance(o, bytes):
            return six.ensure_str(o)
        return json.JSONEncoder.default(self, o)


def dump_numpy(data, fp, pretty=False, **kwargs):  # pylint: disable=invalid-name
    """ Dump object to json, converting numpy objects to reasonable JSON """
    if pretty:
        json.dump(
            data,
            fp,
            cls=NumpyAwareJSONEncoder,
            indent=kwargs.pop("indent", 4),
            sort_keys=kwargs.pop("sort_keys", True),
            separators=kwargs.pop("separators", (",", ":")),
            **kwargs,
        )
    else:
        json.dump(data, fp, cls=NumpyAwareJSONEncoder, **kwargs)
