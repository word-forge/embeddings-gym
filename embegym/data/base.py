import pickle
import collections.abc
import abc

from embegym.utils.io import save_meta, try_load_meta


class BaseCorpus(object):
    def __iter__(self):
        return iter(self._generate_samples())

    def __len__(self):
        return self.meta.get('samples_count', None)

    @abc.abstractmethod
    def _generate_samples(self):
        pass


class BaseFileCorpus(BaseCorpus):
    def __init__(self, filename):
        self.filename = filename
        self.meta = try_load_meta(self.filename)


class LineCorpus(BaseFileCorpus):
    def __init__(self, filename, delimiter=' '):
        super(LineCorpus, self).__init__(filename)
        self.delimiter = delimiter

    def _generate_samples(self):
        with open(self.filename, 'r') as f:
            for line in f:
                yield line.strip().split(self.delimiter)


class PickleCorpus(BaseFileCorpus):
    def _generate_samples(self):
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    return


def cache_corpus_to_pickle(corpus, target_file):
    samples_count = 0
    with open(target_file, 'wb') as f:
        for sample in corpus:
            pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)
            samples_count += 1
    save_meta(dict(samples_count=samples_count), target_file)


def iter_sliding_window(seq, left_ctx_size, right_ctx_size):
    for i, current in enumerate(seq):
        ctx = []
        ctx.extend(seq[i - left_ctx_size : i])
        ctx.extend(seq[i + 1 : i + right_ctx_size + 1])
        yield i, current, ctx


class SlidingWindow(BaseCorpus):
    def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2):
        assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
        self.base_corpus = base_corpus
        self.left_ctx_size = left_ctx_size
        self.right_ctx_size = right_ctx_size

    def _generate_samples(self):
        for sample_elems in self.base_corpus:
            for _, current, ctx in iter_sliding_window(sample_elems,
                                                       self.left_ctx_size,
                                                       self.right_ctx_size):
                yield dict(current=current,
                           context=ctx)


class SlidingWindowAndGlobal(BaseCorpus):
    def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2):
        assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
        self.base_corpus = base_corpus
        self.left_ctx_size = left_ctx_size
        self.right_ctx_size = right_ctx_size

    def _generate_samples(self):
        for sample_elems in self.base_corpus:
            for _, current, ctx in iter_sliding_window(sample_elems,
                                                       self.left_ctx_size,
                                                       self.right_ctx_size):
                yield dict(current=current,
                           context=ctx,
                           global_context=list(sample_elems))
