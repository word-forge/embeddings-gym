import logging
import abc
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


from ..base import TrainableModel
from ..vocab import Vocabulary
from embegym.utils.net.base import run_network_over_data, np2tensor, module_on_cuda
from embegym.utils import copy_with_prefix
from embegym.utils.io import save_meta, try_load_meta



logger = logging.getLogger()


def _default_init(module):
    classname = module.__class__.__name__
    if 'BatchNorm' in classname:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    else:
        module.weight.data.normal_(0.0, 1.0)


def const_lr(value=1e-3):
    return value


class PytorchBaseModel(TrainableModel):
    def __init__(self,
                 vector_size=300,
                 vocabulary=None,
                 criterion=F.binary_cross_entropy,
                 epochs=10,
                 batch_size=128,
                 optimizer_cls=torch.optim.SparseAdam,
                 lr_getter=const_lr,
                 vocab_max_words_final=300000,
                 vocab_max_words_training=1000000,
                 vocab_min_count=5,
                 cuda=torch.cuda.is_available(),
                 verbose=1):
        self._trained = False
        self._vector_size = vector_size
        self._vocabulary = vocabulary
        self._criterion = criterion
        self._epochs = epochs
        self._batch_size = batch_size
        self._optimizer_cls = optimizer_cls
        self._lr_getter = lr_getter
        self._vocab_max_words_final = vocab_max_words_final
        self._vocab_max_words_training = vocab_max_words_training
        self._vocab_min_count = vocab_min_count
        self._cuda = cuda
        self._verbose = verbose
        self._model = None

    @classmethod
    def load(cls, path, *args, **kwargs):
        result = cls()
        vars(result).update(try_load_meta(path))
        result._model = torch.load(path)
        return result

    def save(self, path):
        model = self._model
        self._model = None
        torch.save(model, path)
        save_meta(vars(self), path)

    def reinitialize(self):
        self._model.apply(_default_init)

    def train(self, data, val_data=None, update_existing_vocabulary=False):
        if self._vocabulary is None:
            new_vocab = True
            self._vocabulary = self._create_vocabulary()
        else:
            new_vocab = False

        if new_vocab or update_existing_vocabulary:
            for sample in data:
                self._update_vocab(sample)
            self._vocabulary.finalize()

        if self._model is None:
            self._model = self._make_model()
            if self._cuda:
                self._model.cuda()
            else:
                self._model.cpu()

        history = []

        for epoch_i in range(self._epochs):
            if self._verbose > 0:
                logger.info('Epoch {}', epoch_i)
            optimizer = self._optimizer_cls(self._model.parameters(),
                                            lr=self._lr_getter(epoch_i))

            train_metrics = run_network_over_data(self._make_batch_generator(data),
                                                  self._model,
                                                  self._criterion,
                                                  optimizer=optimizer,
                                                  verbose=self._verbose)
            cur_hist = dict()
            copy_with_prefix(cur_hist, train_metrics, 'train_')

            if self._verbose > 0:
                logger.info('Train loss', epoch_i)
            if val_data:
                val_metrics = run_network_over_data(data,
                                                    self._model,
                                                    self._criterion,
                                                    verbose=self._verbose)
                copy_with_prefix(cur_hist, val_metrics, 'val_')

            history.append(cur_hist)

    def _create_vocabulary(self):
        return Vocabulary(max_words_final=self._vocab_max_words_final,
                          min_count=self._vocab_min_count,
                          truncate_at=self._vocab_max_words_training)

    def _update_vocab(self, sample):
        self._vocabulary.update(sample['current'])

    @abc.abstractmethod
    def _make_model(self):
        pass

    @abc.abstractmethod
    def _make_batch_generator(self, data):
        pass

    def __contains__(self, word):
        assert self._trained
        return word in self._vocabulary

    def vector_size(self):
        return self._vector_size

    def export(self):
        return (list(self._vocabulary.idx2word),
                [self._model.main_embeddings.data.cpu().numpy()])

    def known_words(self):
        return list(self._vocabulary.idx2word)

    def get_word_vector(self, word, *args, **kwargs):
        assert word in self._vocabulary
        return self._model.main_embeddings[self._vocabulary[word].idx].data.cpu().numpy()

    def get_most_similar(self, vector, k=10, *args, **kwargs):
        vector = np2tensor(vector, cuda=module_on_cuda(self))
        probs = torch.matmul(self._model.main_embeddings.data, vector).cpu().numpy()
        best_idx = numpy.argpartition(-probs, k)[:k]
        result = [(self._vocabulary.idx2word[i], probs[i]) for i in best_idx]
        result.sort(key=lambda p: -p[1])
        return result
