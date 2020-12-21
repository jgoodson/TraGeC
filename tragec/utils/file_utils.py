"""
From songlab-cal TAPE: https://github.com/songlab-cal/tape
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import argparse

import os
import random

import typing

from time import strftime, gmtime


def int_or_str(arg: str) -> typing.Union[int, str]:
    try:
        return int(arg)
    except ValueError:
        return arg


def check_is_file(file_path: typing.Optional[str]) -> typing.Optional[str]:
    if file_path is None or os.path.isfile(file_path):
        return file_path
    else:
        raise argparse.ArgumentTypeError(f"File path: {file_path} is not a valid file")


def check_is_dir(dir_path: typing.Optional[str]) -> typing.Optional[str]:
    if dir_path is None or os.path.isdir(dir_path):
        return dir_path
    else:
        raise argparse.ArgumentTypeError(f"Directory path: {dir_path} is not a valid directory")


def get_expname(exp_name: typing.Optional[str],
                task: typing.Optional[str] = None,
                model_type: typing.Optional[str] = None) -> str:
    if exp_name is None:
        time_stamp = strftime("%y-%m-%d-%H-%M-%S", gmtime())
        exp_name = f"{task}_{model_type}_{time_stamp}_{random.randint(0, int(1e6)):0>6d}"
    return exp_name
