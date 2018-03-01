# cython: profile=False

import pickle
import collections.abc
import abc

from embegym.utils.io import save_meta, try_load_meta
from embegym.utils import get_tqdm


class BaseCorpus(object):
    meta = {}

    def __init__(self, verbose=1):
        self._verbose = verbose

    def __iter__(self):
        for s in self._generate_samples_outer():
            yield s

    def __len__(self):
        return self.meta.get('samples_count', 0)

    def _generate_samples_outer(self):
        gen = self._generate_samples()
        if self._verbose > 0:
            cur_len = len(self)
            if cur_len is None:
                return get_tqdm(gen)
            else:
                return get_tqdm(gen, total=cur_len)
        else:
            return gen

    @abc.abstractmethod
    def _generate_samples(self):
        pass


class BaseFileCorpus(BaseCorpus):
    def __init__(self, filename, verbose=1):
        super(BaseFileCorpus, self).__init__(verbose=verbose)
        self.filename = filename
        self.meta = try_load_meta(self.filename)


class LineCorpus(BaseFileCorpus):
    def __init__(self, filename, delimiter=' ', verbose=1):
        super(LineCorpus, self).__init__(filename, verbose=verbose)
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


class LimitedCorpus(BaseCorpus):
    def __init__(self, base, limit=1000, verbose=1):
        super(LimitedCorpus, self).__init__(verbose=verbose)
        self.samples = []
        for i, s in enumerate(base):
            if i >= limit:
                break
            self.samples.append(s)
        self.meta['samples_count'] = len(self.samples)

    def _generate_samples(self):
        for s in self.samples:
            yield s


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
    def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2, verbose=1):
        assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
        super(SlidingWindow, self).__init__(verbose=verbose)
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
    def __init__(self, base_corpus, left_ctx_size=2, right_ctx_size=2, verbose=1):
        assert isinstance(next(iter(base_corpus)), collections.abc.Sequence)
        super(SlidingWindowAndGlobal, self).__init__(verbose=verbose)
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
