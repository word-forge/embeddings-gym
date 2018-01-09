import torch
import torch.nn as nn
import numpy

from .base import PytorchBaseModel
from embegym.utils.net.base import collect_batch


def _unsqueeze_word_idx(words):
    dims = len(words.size())
    if dims == 1:
        return words.unsqueeze(-1)
    elif dims == 2:
        return words
    else:
        raise Exception()


class _SkipGramModel(nn.Module):
    def __init__(self, word_number, vector_size):
        super(_SkipGramModel, self).__init__()
        self.main_embeddings = nn.Embedding(vector_size,
                                            word_number)
        self.ctx_embeddings = nn.Embedding(vector_size,
                                           word_number)

    def forward(self, word_pairs):
        main_words, ctx_words = word_pairs
        main_vectors = self.main_embeddings(_unsqueeze_word_idx(main_words))  # (B, 1, EmbSize)
        main_norm = main_vectors.norm(dim=1)
        ctx_vectors = self.ctx_embeddings(_unsqueeze_word_idx(ctx_words))  # (B, 1, EmbSize)
        ctx_norm = ctx_vectors.norm(dim=1)
        return (main_vectors * ctx_vectors).sum(1) / (main_norm * ctx_norm)


class UniformNegativeSampler(object):
    def __init__(self, vocab_size, words_to_sample):
        self.vocab_size = vocab_size
        self.words_to_sample = words_to_sample

    def __call__(self, positive_words_idx):
        result = set()
        while len(result) < self.words_to_sample:
            idx = numpy.random.randint(0, self.vocab_size, self.words_to_sample - len(result))
            for i in idx:
                if i not in positive_words_idx and i not in result:
                    result.add(i)
        return result


def skipgram_negative_sampling_generator(data, vocab, negative_sampler, prefix_modality=False):
    for sample in data:
        cur_word = sample['current']
        cur_idx = vocab[cur_word].idx
        del sample['current']
        real_neighbors_idx = {vocab[(mod_name + '__' + w) if prefix_modality else w].idx
                              for mod_name, neighbors in data.items()
                              for w in neighbors}
        for ctx_idx in real_neighbors_idx:
            yield (cur_idx, ctx_idx), 1
            for neg_idx in negative_sampler(real_neighbors_idx):
                yield (cur_idx, neg_idx), 0


class Word2VecSgNs(PytorchBaseModel):
    def __init__(self, negative=5, *args, **kwargs):
        super(Word2VecSgNs, self).__init__(*args, **kwargs)
        self._negative = negative

    def _make_model(self):
        return _SkipGramModel(len(self._vocabulary), self.vector_size())

    def _make_batch_generator(self, data):
        return collect_batch(skipgram_negative_sampling_generator(data,
                                                                  self._vocabulary,
                                                                  UniformNegativeSampler(len(self._vocabulary))),
                             self._batch_size)
