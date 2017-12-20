import os


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
                    with open(p, 'r') as f:
                        return f.read()
                else:
                    return 'file', p

    return None
