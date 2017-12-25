import sklearn.datasets
import torch

from embegym.data.tokenization import DEFAULT_TOKENIZER
from .base import MeanEmbeddingLogregEvaluatorBase, CNNBiLstmClsEvaluatorBase


_PREPARED_NEWSGROUPS = None


def prepare_20newsgroups():
    global _PREPARED_NEWSGROUPS
    if _PREPARED_NEWSGROUPS is None:
        train = sklearn.datasets.fetch_20newsgroups(subset='train',
                                                    remove=('headers', 'footers', 'quotes'))
        x_train = [DEFAULT_TOKENIZER(txt)[0] for txt in train.data]

        test = sklearn.datasets.fetch_20newsgroups(subset='test',
                                                   remove=('headers', 'footers', 'quotes'))
        x_test = [DEFAULT_TOKENIZER(txt)[0] for txt in test.data]
        _PREPARED_NEWSGROUPS = [(x_train, train.target, x_test, test.target)]
    return _PREPARED_NEWSGROUPS


class MeanEmbeddingLogreg20NG(MeanEmbeddingLogregEvaluatorBase):
    def __init__(self, *args, **kwargs):
        super(MeanEmbeddingLogreg20NG, self).__init__(prepare_20newsgroups(), *args, **kwargs)


class CnnLstm20NG(CNNBiLstmClsEvaluatorBase):
    def __init__(self, *args, **kwargs):
        data = prepare_20newsgroups()
        labels = set(data[0][1]) | set(data[0][3])
        super(CnnLstm20NG, self).__init__(data,
                                          out_classes=len(labels),
                                          *args, **kwargs)
