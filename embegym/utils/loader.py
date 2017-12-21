import os
import importlib
import smart_open


BASE_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  '_resources')
RESOURCES_PATHS = [
    BASE_RESOURCES_DIR
]


def try_get_resource(identifier, read=True):
    if isinstance(identifier, str):
        for base_path in RESOURCES_PATHS:
            p = os.path.join(base_path, identifier)
            if os.path.isfile(p):
                if read:
                    with smart_open.smart_open(p, 'r') as f:
                        return f.read()
                else:
                    return p

    return None


def get_fully_qualified_name(cls):
    return cls.__module__ + '.' + cls.__name__


def get_fully_qualified_class_name(obj):
    return get_fully_qualified_name(type(obj))


def load_class(name):
    if '.' in name:
        module_name, name = name.rsplit('.', 1)
        ctx = importlib.import_module(module_name)
    else:
        ctx = globals()
    return getattr(ctx, name)
