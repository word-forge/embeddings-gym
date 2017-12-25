class Registry(object):
    def __init__(self, **objects):
        self._objects = dict(objects)

    def register(self, name, obj):
        assert name not in self._objects
        self._objects[name] = obj

    def objects(self, names):
        if not names:
            names = self._objects.keys()
        return list((name, self._objects[name]) for name in names)

    def names(self):
        return list(self._objects.keys())