import tqdm
from . import io, loader, features, registry, net, logger


def copy_with_prefix(to_dict, from_dict, prefix):
    to_dict.update((prefix + k, v)
                   for k, v in from_dict.items())


def is_in_jupyter():
    try:
        get_ipython
        return True
    except:
        return False


def get_tqdm(*args, **kwargs):
    if is_in_jupyter():
        return tqdm.tqdm_notebook(*args, **kwargs)
    return tqdm.tqdm(*args, **kwargs)
