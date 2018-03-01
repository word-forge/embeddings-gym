# cython: profile=False

import numpy
cimport numpy

import torch
from torch.utils.data.dataloader import default_collate


def word_pairs_outcome_collate(list lst):
    cdef int li, ri, row
    cdef float out
    cdef numpy.ndarray[numpy.int64_t] left_idx = numpy.zeros(len(lst), dtype='int64')
    cdef numpy.ndarray[numpy.int64_t] right_idx = numpy.zeros(len(lst), dtype='int64')
    cdef numpy.ndarray[numpy.float32_t] outcomes = numpy.zeros(len(lst), dtype='float32')
    for row, ((li, ri), out) in enumerate(lst):
        left_idx[row] = li
        right_idx[row] = ri
        outcomes[row] = out
    return (torch.from_numpy(left_idx), torch.from_numpy(right_idx)), torch.from_numpy(outcomes)


def collect_batch(gen, batch_size, collate_func=default_collate):
    batch = []
    for sample in gen:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield collate_func(batch)
            batch = []
    if len(batch) > 0:
        yield collate_func(batch)


def infinite_iter(base_iter):
    while True:
        for elem in base_iter:
            yield elem
