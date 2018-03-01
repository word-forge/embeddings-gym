# cython: profile=False

import numpy
cimport numpy
cimport cython


cdef class CachedRandInt:
    cdef int low, high, cache_size, i
    cdef numpy.int64_t[:] cache

    def __init__(self, low, high, cache_size=1000000):
        self.low = low
        self.high = high
        self.cache_size = cache_size
        self.recharge()

    @cython.profile(False)
    cdef recharge(self):
        self.cache = numpy.random.randint(self.low, self.high, self.cache_size)
        self.i = 0

    @cython.profile(False)
    cpdef get(self):
        if self.i >= self.cache_size:
            self.recharge()
        res = self.cache[self.i]
        self.i += 1
        return res

    @cython.profile(False)
    cpdef gen_set(self, int n, set to_be_excluded):
        cdef int i, x
        cdef list result = []
        for i in range(n):
            x = self.get()
            if x not in to_be_excluded:
                result.append(x)
        return result


cdef class UniformNegativeSampler(object):
    cdef CachedRandInt gen
    cdef int words_to_sample

    def __init__(self, int vocab_size, int words_to_sample):
        self.gen = CachedRandInt(0, vocab_size)
        self.words_to_sample = words_to_sample

    def __call__(self, set positive_words_idx):
        return self.gen.gen_set(self.words_to_sample, positive_words_idx)


def skipgram_negative_sampling_generator(data, vocab, negative_sampler, prefix_modality=False):
    cdef int cur_idx, neg_idx
    for sample in data:
        cur_word = sample['current']

        if cur_word not in vocab:
            continue

        cur_idx = vocab[cur_word].idx
        del sample['current']
        if prefix_modality:
            neighbors = {mod_name + '__' + w
                         for mod_name, neighbors in sample.items()
                         for w in neighbors}
        else:
            neighbors = {w
                         for mod_name, neighbors in sample.items()
                         for w in neighbors}
        real_neighbors_idx = {vocab[w].idx for w in neighbors if w in vocab}
        for ctx_idx in real_neighbors_idx:
            yield (cur_idx, ctx_idx), 1.0
            for neg_idx in negative_sampler(real_neighbors_idx):
                yield (cur_idx, neg_idx), 0.0