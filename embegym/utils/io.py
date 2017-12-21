import os
import urllib.request
import pickle


def default_save(obj, fname, protocol=pickle.HIGHEST_PROTOCOL):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def default_load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


METAFILE_SUFFIX = '.embegym'


def make_metafile_path(fname):
    return fname + METAFILE_SUFFIX


def save_meta(data, base_path):
    default_save(data, make_metafile_path(base_path))


def try_load_meta(base_path, default={}):
    try:
        return default_load(make_metafile_path(base_path))
    except IOError:
        return default


def download_if_not_exists(uri, to_file):
    if not os.path.exists(to_file):
        urllib.request.urlretrieve(uri, to_file)
