from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText
from .base import BaseModel, TrainableModel
from embegym.utils.io import save_meta, try_load_meta
from embegym.utils.loader import load_class, get_fully_qualified_class_name, get_fully_qualified_name


class GensimKeyedVectorsMixin(object):
    """
    Simple wrapper for gensim.models.KeyedVectors
    """

    DEFAULT_IMPL_CLASSNAME = get_fully_qualified_name(KeyedVectors)

    def __init__(self, _impl):
        self._impl = _impl

    @classmethod
    def load(cls, path, *args, **kwargs):
        meta_info = try_load_meta(path)
        impl_classname = meta_info.get('impl_class', cls.DEFAULT_IMPL_CLASSNAME)
        impl_cls = load_class(impl_classname)
        return cls(impl_cls.load(path, *args, **kwargs))

    def save(self, path):
        save_meta(dict(impl_class=get_fully_qualified_class_name(self._impl)),
                  path)
        self._impl.save(path)

    def get_word_vector(self, word, *args, **kwargs):
        return self._impl[word]

    def __contains__(self, word):
        return word in self._impl.vw.vocab

    def vector_size(self):
        return self._impl.vw.vector_size

    def export(self):
        return self._impl.vw.index2word, [self._impl.vw.syn0]

    def get_most_similar(self, vector, k=10, *args, **kwargs):
        return self._impl.vw.similar_by_vector(vector, topn=k, *args, **kwargs)


class GensimKeyedVectors(GensimKeyedVectorsMixin, BaseModel):
    pass


class GensimBaseTrainableModel(GensimKeyedVectorsMixin, TrainableModel):
    """
    Simple wrapper for gensim.models.Word2Vec
    """
    def __init__(self, gensim_model, **train_kwargs):
        assert hasattr(gensim_model, 'train')
        super(GensimBaseTrainableModel, self).__init__(gensim_model)
        self._train_kwargs = train_kwargs

    def reinitialize(self):
        self._impl.reset_weights()

    def train(self, data):
        self._impl.train(data, **self._train_kwargs)


def import_pretrained_word2vec(in_file, out_file, mmap='r'):
    try:
        return GensimKeyedVectorsMixin.load(out_file, mmap=mmap)
    except IOError:
        pass
    model = GensimKeyedVectorsMixin(KeyedVectors.load_word2vec_format(in_file))
    model.save(out_file)
    return GensimKeyedVectorsMixin.load(out_file, mmap=mmap)


def import_pretrained_fasttext(in_file, out_file, mmap='r'):
    try:
        return GensimKeyedVectorsMixin.load(out_file, mmap=mmap)
    except IOError:
        pass
    model = GensimKeyedVectorsMixin(FastText.load_fasttext_format(in_file))
    model.save(out_file)
    return GensimKeyedVectorsMixin.load(out_file, mmap=mmap)
