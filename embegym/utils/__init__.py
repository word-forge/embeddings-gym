from . import io, loader, features, registry, net


def copy_with_prefix(to_dict, from_dict, prefix):
    to_dict.update((prefix + k, v)
                   for k, v in from_dict.items())
