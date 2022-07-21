#!/usr/bin/env python
#
# Copyright (c) 2014 10X Genomics, Inc. All rights reserved.
#

""" Helper for functions for logging calls to subprocesses.

In addition to logging, these wrappers will also (on linux) set PDEATHSIG on
the child process, so that they are killed if the parent process dies. """


from __future__ import absolute_import

import subprocess
import logging
import sys

from typing import (  # pylint: disable=import-error,unused-import
    Any,
    AnyStr,
    IO,
    List,
    Tuple,
    Union,
)

from six import ensure_str

# On linux, provide a method to set PDEATHSIG on child processes.
if sys.platform.startswith("linux"):
    import ctypes
    import ctypes.util
    from signal import SIGKILL

    _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))  # type: ctypes.CDLL

    _PR_SET_PDEATHSIG = ctypes.c_int(1)  # type: ctypes.c_int # <sys/prctl.h>

    def child_preexec_set_pdeathsig():
        """When used as the preexec_fn argument for subprocess.Popen etc,
        causes the subprocess to recieve SIGKILL if the parent process
        terminates."""
        zero = ctypes.c_ulong(0)
        _LIBC.prctl(_PR_SET_PDEATHSIG, ctypes.c_ulong(SIGKILL), zero, zero, zero)


else:
    child_preexec_set_pdeathsig = None  # pylint: disable=invalid-name


def _str_args(args):
    # type: (Union[List[AnyStr], Tuple[AnyStr]]) -> List[str]
    if isinstance(args, (list, tuple)):
        return [ensure_str(x) for x in args]
    else:
        raise ValueError("Command must be a list or tuple.")


def check_call(args, *wargs, **kwargs):
    # type: (List[AnyStr], Any, Any) -> int
    """ See `subprocess.check_call`. """
    # pylint: disable=no-value-for-parameter
    logging.log(logging.INFO, "subprocess check_call: %s", " ".join(_str_args(args)))
    if "preexec_fn" not in kwargs:
        kwargs["preexec_fn"] = child_preexec_set_pdeathsig
    return subprocess.check_call(args, *wargs, **kwargs)


def check_output(args, *wargs, **kwargs):
    # type: (...) -> bytes
    """ See `subprocess.check_output`. """
    # pylint: disable=no-value-for-parameter
    logging.log(logging.INFO, "subprocess check_output: %s", " ".join(_str_args(args)))
    if "preexec_fn" not in kwargs:
        kwargs["preexec_fn"] = child_preexec_set_pdeathsig
    return subprocess.check_output(args, *wargs, **kwargs)


def call(args, *wargs, **kwargs):
    # type: (List[AnyStr], Any, Any) -> int
    """ See `subprocess.call`. """
    # pylint: disable=no-value-for-parameter
    logging.log(logging.INFO, "subprocess call: %s", " ".join(_str_args(args)))
    if "preexec_fn" not in kwargs:
        kwargs["preexec_fn"] = child_preexec_set_pdeathsig
    return subprocess.call(args, *wargs, **kwargs)


def Popen(args, *wargs, **kwargs):  # pylint: disable=invalid-name
    # type: (List[AnyStr], Any, Any) -> subprocess.Popen[bytes]
    """ See `subprocess.Popen`. """
    # pylint: disable=no-value-for-parameter
    logging.log(logging.INFO, "subprocess Popen: %s", " ".join(_str_args(args)))
    if "preexec_fn" not in kwargs:
        kwargs["preexec_fn"] = child_preexec_set_pdeathsig
    return subprocess.Popen(args, *wargs, **kwargs)


def pipeline(cmds, stdin=None, close_input=False, stdout=None, close_output=False, log=True):
    # type: (List[List[AnyStr]], Union[None, str, bytes, IO[bytes]], bool, Union[None, str, bytes, IO[bytes]], bool, bool) -> None
    """Run a series of commands, piping output from each one to the next.

    The commands given as arguments to this method must be lists of arguments.
    In general, this will be much safer than using `check_call` with `shell=True`
    to pipe output between processes.  For example,

    .. code-block:: python

       from subprocess import call
       filename = input("What file would you like to search?\\n")
       # imagine the user enters `non_existent; rm -rf / #`
       call("cat {} | grep predicate".format(filename), shell=True)  # Uh oh...

    While it _is_ possible to make sure the arguments in such an expression are
    properly quoted, there are a lot of tricky edge cases to watch out for.

    As compared to `pipes.Template()`, this method makes it easier to safely
    include commands in the pipeline which include untrusted arguments, and also
    offers more flexibility in terms of inputs and outputs.

    Args:
        cmds (list of list): A list of commands to execute.  Must have at least
                             one element.
        stdin (file or str): A filename or file object to pass as input to the
                             first process in the pipeline.  This file will be
                             closed once the subprocess is launched.  If not
                             provided, input will not be redirected.
        close_input (bool): If true, close the input file after launching the
                            first process.  This value is ignored if
                            `input_file` is not a file-like object.
        stdout (file or str): The name of the file to which to write the output
                              of the final command in the pipeline.  If not
                              provided, the final process's output will not be
                              redirected.
        close_output (bool): If true, close the output file after launching the
                             final process.  This value is ignored if
                             `output_file` is not a file-like object.
        log (bool): If true, write the pipeline (as if it were a bash command)
                    to the log.
    """
    assert isinstance(cmds, list)
    assert len(cmds) > 0
    assert all(isinstance(cmd, list) for cmd in cmds)

    if log:
        logging.log(
            logging.INFO,
            "subprocess Popen: %s",
            " | ".join(" ".join(_str_args(cmd)) for cmd in cmds),
        )
    # This call goes recursive but we don't want to log every command in the
    # pipeline.
    _pipeline(
        cmds,
        input_file=stdin,
        close_input=close_input,
        output_file=stdout,
        close_output=close_output,
    )


def _pipeline(cmds, input_file, close_input, output_file, close_output):
    # type: (List[List[AnyStr]], Union[None, str, bytes, IO[bytes]], bool, Union[None, str, bytes, IO[bytes]], bool) -> None
    # This is mostly mimiking the implementation of subprocess.call()
    # https://github.com/python/cpython/blob/v3.7.7/Lib/subprocess.py#L331
    proc = _get_pipeline_proc(
        cmds[0], input_file, close_input, output_file, close_output, len(cmds) == 1
    )
    try:
        try:
            if len(cmds) > 1:
                # Recurse to run the next process in the pipeline.
                _pipeline(
                    cmds[1:],
                    input_file=proc.stdout,
                    close_input=True,
                    output_file=output_file,
                    close_output=close_output,
                )
        except:
            proc.kill()
            raise
    finally:
        result = proc.wait()
    if result:
        raise subprocess.CalledProcessError(result, cmds[0])


def _is_file_handle(obj):
    # type: (Any) -> bool
    return hasattr(obj, "fileno") and hasattr(obj, "close")


def _get_pipeline_proc(cmd, input_file, close_input, output_file, close_output, final):
    # type: (...) -> subprocess.Popen[bytes]
    """Starts a command with the given input and output.

    Handles all of the details around opening files and closing them if need be,
    even when stuff fails.
    """
    if input_file and not _is_file_handle(input_file):
        input_file = open(input_file, "rb")
        close_input = True
    proc = None
    try:
        if not final:
            proc = subprocess.Popen(  # pylint: disable=bad-option-value,subprocess-popen-preexec-fn
                cmd,
                stdin=input_file,
                stdout=subprocess.PIPE,
                preexec_fn=child_preexec_set_pdeathsig,
            )
            return proc
        if output_file and not _is_file_handle(output_file):
            output_file = open(output_file, "wb")
            close_output = True
        try:
            proc = subprocess.Popen(  # pylint: disable=bad-option-value,subprocess-popen-preexec-fn
                cmd, stdin=input_file, stdout=output_file, preexec_fn=child_preexec_set_pdeathsig,
            )
            return proc
        finally:
            if output_file and close_output and _is_file_handle(output_file):
                try:
                    output_file.close()
                except:
                    logging.log(logging.INFO, "failed to close output file")
                    if proc:
                        proc.kill()
                        proc.wait()
                    raise
    finally:
        if input_file and close_input and _is_file_handle(input_file):
            try:
                input_file.close()
            except:
                if proc:
                    proc.kill()
                    proc.wait()
                raise
