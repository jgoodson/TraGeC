"""
Utilities for working with the local datamodule cache.
This file is adapted from the huggingface transformers library at
https://github.com/huggingface/transformers, which in turn is adapted from the AllenNLP
library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
Note - this file goes to effort to support Python 2, but the rest of this repository does not.

From songlab-cal TAPE: https://github.com/songlab-cal/tape
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import argparse
import fnmatch
import json
import os
import random
import sys
import tempfile
import typing
from contextlib import contextmanager
from functools import partial, wraps
from hashlib import sha256
from io import open
from time import strftime, gmtime
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError
from filelock import FileLock
from tqdm import tqdm

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    _get_torch_home = None
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'protein_models')

try:
    from pathlib import Path

    PYTORCH_PRETRAINED_BERT_CACHE: typing.Union[str, Path] = Path(
        os.getenv('PROTEIN_MODELS_CACHE', os.getenv(
            'PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path)))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PROTEIN_MODELS_CACHE',
                                              os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                        default_cache_path))

PROTEIN_MODELS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility


def get_etag(url):
    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        try:
            response = requests.head(url, allow_redirects=True)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get("ETag")
        except EnvironmentError:
            etag = None

    return etag


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def cached_path(url_or_filename, force_download=False, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.

    Args:
        url_or_filename: string containing the location of the file to find in cache
        cache_dir: specify a cache directory to save the file to
        (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's
        already cached in the cache dir.
    """
    if cache_dir is None:
        cache_dir = PROTEIN_MODELS_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(url_or_filename, cache_dir, force_download)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(
            url_or_filename))

    return output_path


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None, force_download=False, resume_download=False):
    """
    Given a URL, look for the corresponding datamodule in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PROTEIN_MODELS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    etag = get_etag(url)

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and etag is None:
        return cache_path

    # If we don't have a connection (etag is None) and can't identify the file
    # try to get the last downloaded one
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
        matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])

    # From now on, etag is not None
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir,
                                        delete=False)
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            http_get(url, temp_file)

        os.replace(temp_file.name, cache_path)

        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


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
