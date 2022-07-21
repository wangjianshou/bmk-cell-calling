#!/usr/bin/env python
#
# Copyright (c) 2015 10X Genomics, Inc. All rights reserved.
#

from __future__ import absolute_import, print_function
import os
import errno
import shutil
import subprocess
import sys
import gzip
import _io as io  # this is necessary b/c this module is named 'io' ... :(
from six import ensure_binary, string_types, PY3
import lz4.frame as lz4
import bmkcc.tenkit.log_subprocess as tk_subproc
import bmkcc.cellranger.h5_constants as h5_constants

if PY3:
    from typing import AnyStr  # pylint: disable=unused-import


def fixpath(path):
    # type: (str) -> str
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def get_input_path(oldpath, is_dir=False):
    if not isinstance(oldpath, string_types):
        sys.exit("'{}' is not a valid string type and is an invalid path".format(oldpath))
    path = fixpath(oldpath)
    if not os.path.exists(path):
        sys.exit("Input file does not exist: %s" % path)
    if not os.access(path, os.R_OK):
        sys.exit("Input file path {} does not have read permissions".format(path))
    if is_dir:
        if not os.path.isdir(path):
            sys.exit("Please provide a directory, not a file: %s" % path)
    else:
        if not os.path.isfile(path):
            sys.exit("Please provide a file, not a directory: %s" % path)
    return path


def get_input_paths(oldpaths):
    paths = []
    for oldpath in oldpaths:
        paths.append(get_input_path(oldpath))
    return paths


def get_output_path(oldpath):
    path = fixpath(oldpath)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        sys.exit("Output directory does not exist: %s" % dirname)
    if not os.path.isdir(dirname):
        sys.exit("Please provide a directory, not a file: %s" % dirname)
    return path


def open_maybe_gzip(filename, mode="r"):
    # type: (AnyStr, str) -> io.IOBase
    # this _must_ be a bytes
    if not isinstance(filename, bytes):
        filename = (
            ensure_binary(filename)
            if isinstance(filename, string_types)
            else ensure_binary(str(filename))
        )
    if filename.endswith(h5_constants.GZIP_SUFFIX):
        raw = gzip.open(filename, mode, 2)
    elif filename.endswith(h5_constants.LZ4_SUFFIX):
        raw = lz4.open(filename, mode)
    else:
        return open(filename, mode)

    bufsize = 1024 * 1024  # 1MB of buffering
    if mode == "r":
        return io.TextIOWrapper(io.BufferedReader(raw, buffer_size=bufsize))
    elif mode == "w":
        return io.TextIOWrapper(io.BufferedWriter(raw, buffer_size=bufsize))
    elif mode == "rb":
        return io.BufferedReader(raw, buffer_size=bufsize)
    elif mode == "wb":
        return io.BufferedWriter(raw, buffer_size=bufsize)

    else:
        raise ValueError("Unsupported mode for compression: %s" % mode)


class CRCalledProcessError(Exception):
    def __init__(self, msg):
        super(CRCalledProcessError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


def run_command_safely(cmd, args):
    p = tk_subproc.Popen([cmd] + args, stderr=subprocess.PIPE)
    _, stderr_data = p.communicate()
    if p.returncode != 0:
        raise Exception("%s returned error code %d: %s" % (p, p.returncode, stderr_data))


def check_completed_process(p, cmd):
    """p   (Popen object): Subprocess
    cmd (str):          Command that was run
    """
    if p.returncode is None:
        raise CRCalledProcessError("Process did not finish: %s ." % cmd)
    elif p.returncode != 0:
        raise CRCalledProcessError("Process returned error code %d: %s ." % (p.returncode, cmd))


def mkdir(dst, allow_existing=False):
    """Create a directory. Optionally succeed if already exists.
    Useful because transient NFS server issues may induce double creation attempts."""
    if allow_existing:
        try:
            os.mkdir(dst)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(dst):
                pass
            else:
                raise
    else:
        os.mkdir(dst)


def makedirs(dst, allow_existing=False):
    """Create a directory recursively. Optionally succeed if already exists.
    Useful because transient NFS server issues may induce double creation attempts."""
    if allow_existing:
        try:
            os.makedirs(dst)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(dst):
                pass
            else:
                raise
    else:
        os.makedirs(dst)


def remove(f, allow_nonexisting=False):
    """Delete a file. Allow to optionally succeed if file doesn't exist.
    Useful because transient NFS server issues may induce double deletion attempts."""
    if allow_nonexisting:
        try:
            os.remove(f)
        except OSError as e:
            if e.errno == errno.ENOENT:
                pass
            else:
                raise
    else:
        os.remove(f)


def copy(src, dst):
    """ Safely copy a file. Not platform-independent """
    run_command_safely("cp", [src, dst])


def move(src, dst):
    """ Safely move a file. Not platform-independent """
    run_command_safely("mv", [src, dst])


def copytree(src, dst, allow_existing=False):
    """ Safely recursively copy a directory. Not platform-independent """
    makedirs(dst, allow_existing=allow_existing)

    for name in os.listdir(src):
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)

        if os.path.isdir(srcname):
            copytree(srcname, dstname)
        else:
            copy(srcname, dstname)


def hardlink_with_fallback(src, dst):
    """Hard-links src to dst, falling back to copy if it fails.

    If `src` is a directory, it will attempt to recursively hardlink.
    """
    if os.path.isdir(src):
        # recursively hardlink a path, fallback to copy if fail
        try:
            shutil.copytree(src, dst, copy_function=os.link)
        except:
            shutil.copytree(src, dst)
    else:
        # hardlink a file, fallback to copy if fail
        try:
            os.link(src, dst)
        except:
            shutil.copy(src, dst)


def concatenate_files(out_path, in_paths, mode=""):
    with open(out_path, "w" + mode) as out_file:
        for in_path in in_paths:
            with open(in_path, "r" + mode) as in_file:
                shutil.copyfileobj(in_file, out_file)


def concatenate_headered_files(out_path, in_paths, mode=""):
    """Concatenate files, taking the first line of the first file
    and skipping the first line for subsequent files.
    Asserts that all header lines are equal."""
    with open(out_path, "w" + mode) as out_file:
        if len(in_paths) > 0:
            # Write first file
            with open(in_paths[0], "r" + mode) as in_file:
                header = in_file.readline()
                out_file.write(header)
                shutil.copyfileobj(in_file, out_file)

        # Write remaining files
        for in_path in in_paths[1:]:
            with open(in_path, "r" + mode) as in_file:
                this_header = in_file.readline()
                assert this_header == header
                shutil.copyfileobj(in_file, out_file)


def write_empty_json(filename):
    with open(filename, "wb") as f:
        f.write(b"{}")


def touch(path):
    # type: (str) -> None
    """Create an empty file"""
    fh = open(path, "w")
    fh.close()
