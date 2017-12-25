import numpy
import skorch
import torch
import collections


def calc_mean_embedding_for_docs(samples, model):
    result = numpy.zeros((len(samples), model.vector_size()),
                         dtype='float32')
    for sample_i, tokenized_text in enumerate(samples):
        known_vectors = [model.get_word_vector(w)
                         for w in tokenized_text
                         if w in model]
        if len(known_vectors) > 0:
            result[sample_i] = numpy.mean(known_vectors, axis=0)
    return result


def tensorize_sequence(tokenized_text, model, max_len=None):
    known_words = [w for w in tokenized_text if w in model]
    if max_len is None:
        return numpy.array([model.get_word_vector(w)
                            for w in known_words])
    else:
        result = numpy.zeros((max_len, model.vector_size()),
                             dtype='float32')
        for i, w in enumerate(known_words[:min(max_len, len(known_words))]):
            result[i] = model.get_word_vector(w)
        return result


def prepare_embeddings_sequences(samples, model, max_len=None):
    for tokenized_text in samples:
        yield tensorize_sequence(tokenized_text, model, max_len=max_len)


class EmbeddingsSeqDataset(skorch.dataset.Dataset):
    """
    Yields tensors (EmbeddingSize, MaxLen) or (BatchSize, EmbeddingSize, MaxLen)
    """

    def __init__(self, tokenized_texts, labels, model=None, max_len=None, *args, **kwargs):
        super(EmbeddingsSeqDataset, self).__init__(tokenized_texts, labels, *args, **kwargs)
        self.model = model
        self.max_len = max_len

    def transform(self, x, y):
        if isinstance(y, collections.Collection) and len(y) == 0:
            return super(EmbeddingsSeqDataset, self).transform(x, y)

        if not isinstance(y, collections.abc.Collection):
            x = tensorize_sequence(x,
                                   self.model,
                                   max_len=self.max_len).transpose()
        else:
            x = numpy.array([tensorize_sequence(sample,
                                                self.model,
                                                max_len=self.max_len)
                             for sample in x]).transpose((1, 0, 2))
        result = super(EmbeddingsSeqDataset, self).transform(x, y)
        return result
