import logging
import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchBaseModel
from embegym.utils.net.batch_utils import collect_batch, word_pairs_outcome_collate
from .word2vec_compiled import UniformNegativeSampler, skipgram_negative_sampling_generator


logger = logging.getLogger()


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
        self.main_embeddings = nn.Embedding(word_number,
                                            vector_size,
                                            sparse=True)
        self.ctx_embeddings = nn.Embedding(word_number,
                                           vector_size,
                                           sparse=True)

    def forward(self, word_pairs):
        main_words, ctx_words = word_pairs
        main_vectors = self.main_embeddings(_unsqueeze_word_idx(main_words)).squeeze(1)  # (B, EmbSize)
        # main_norm = main_vectors.norm(dim=1)
        ctx_vectors = self.ctx_embeddings(_unsqueeze_word_idx(ctx_words)).squeeze(1)  # (B, EmbSize)
        # ctx_norm = ctx_vectors.norm(dim=1)
        prod = (main_vectors * ctx_vectors).sum(1)
        return F.sigmoid(prod)


class Word2VecSgNs(PytorchBaseModel):
    def __init__(self, negative=5, *args, **kwargs):
        kwargs['clip_gradients'] = None
        super(Word2VecSgNs, self).__init__(*args, **kwargs)
        self._negative = negative

    def _make_model(self):
        logger.info('Init model')
        return _SkipGramModel(len(self._vocabulary), self.vector_size())

    def _make_batch_generator(self, data):
        return collect_batch(skipgram_negative_sampling_generator(data,
                                                                  self._vocabulary,
                                                                  UniformNegativeSampler(len(self._vocabulary),
                                                                                         self._negative)),
                             self._batch_size,
                             collate_func=word_pairs_outcome_collate)
