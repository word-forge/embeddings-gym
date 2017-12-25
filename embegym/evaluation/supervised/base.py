import abc

import numpy
import pandas
import sklearn.linear_model
import sklearn.metrics
import skorch
import torch

from embegym.utils.features import calc_mean_embedding_for_docs, EmbeddingsSeqDataset
from embegym.utils.net.modules import CnnLSTMAttention
from ..base import BaseEvaluator


def choose_average_method(gold_y):
    return 'macro' if len(set(gold_y)) > 2 else 'binary'


def auto_average(f):
    def _impl(y_true, y_pred):
        return f(y_true, y_pred, average=choose_average_method(y_true))
    return _impl


DEFAULT_CATEGORIZATION_METRICS = (
    ('f1', auto_average(sklearn.metrics.f1_score)),
    ('precision', auto_average(sklearn.metrics.precision_score)),
    ('recall', auto_average(sklearn.metrics.recall_score)),
)


class SupervisedEvaluatorBase(BaseEvaluator):
    def __init__(self,
                 train_test_splits,
                 metrics=DEFAULT_CATEGORIZATION_METRICS):
        self._train_test_splits = train_test_splits
        self._metrics = metrics

    def __call__(self, model):
        metric_values = []
        for x_src_train, y_train, x_src_test, y_test in self._train_test_splits:
            x_train_features = self._prepare_features(x_src_train, model)
            x_test_features = self._prepare_features(x_src_test, model)

            clf = self._make_classifier(model)

            clf.fit(x_train_features, y_train)
            y_train_pred = clf.predict(x_train_features)
            y_test_pred = clf.predict(x_test_features)

            cur_metrics = {}
            cur_metrics.update(('train {}'.format(metric_name), metric_func(y_train, y_train_pred))
                               for metric_name, metric_func in self._metrics)
            cur_metrics.update(('test {}'.format(metric_name), metric_func(y_test, y_test_pred))
                               for metric_name, metric_func in self._metrics)
            metric_values.append(cur_metrics)
        return pandas.DataFrame(metric_values).mean(axis=0).to_dict()

    @abc.abstractmethod
    def _make_classifier(self, model):
        pass

    @abc.abstractmethod
    def _prepare_features(self, src_data, model):
        pass


class MeanEmbeddingEvaluatorBase(SupervisedEvaluatorBase):
    def _prepare_features(self, src_data, model):
        return calc_mean_embedding_for_docs(src_data, model)


class MeanEmbeddingLogregEvaluatorBase(MeanEmbeddingEvaluatorBase):
    def __init__(self, train_test_splits, *args, **kwargs):
        super(MeanEmbeddingLogregEvaluatorBase, self).__init__(train_test_splits,
                                                               *args,
                                                               **kwargs)

    def _make_classifier(self, model):
        return sklearn.linear_model.LogisticRegression()


class EmbeddingSeqEvaluatorBase(SupervisedEvaluatorBase):
    def __init__(self, train_test_splits, max_seq_len=100, batch_size=16, *args, **kwargs):
        super(EmbeddingSeqEvaluatorBase, self).__init__(train_test_splits, *args, **kwargs)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def _prepare_features(self, src_data, model):
        return numpy.array(src_data)


class FlattenedLoss(object):
    def __init__(self, base_loss):
        self._impl = base_loss

    def __call__(self, y_pred, target, *args, **kwargs):
        return self._impl(y_pred, target.squeeze(), *args, **kwargs)


class FlattenedNLLLoss(FlattenedLoss):
    def __init__(self):
        super(FlattenedNLLLoss, self).__init__(torch.nn.NLLLoss())


class CNNBiLstmClsEvaluatorBase(EmbeddingSeqEvaluatorBase):
    def __init__(self, train_test_splits,
                 out_classes=2,
                 criterion=FlattenedNLLLoss,
                 lr=1e-3,
                 num_epochs=20,
                 batch_size=128,
                 use_cuda=True,
                 clip_grad_norm=1.0,
                 verbose=0,
                 *args,
                 **kwargs):
        super(CNNBiLstmClsEvaluatorBase, self).__init__(train_test_splits,
                                                        *args, **kwargs)
        self.out_classes = out_classes
        self.criterion = criterion
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.clip_grad_norm = clip_grad_norm
        self.verbose = verbose

    def _make_classifier(self, model):
        return skorch.NeuralNet(module=CnnLSTMAttention,
                                module__embeddings_size=model.vector_size(),
                                module__out_classes=self.out_classes*2,
                                criterion=self.criterion,
                                optimizer=torch.optim.Adam,
                                lr=self.lr,
                                max_epochs=self.num_epochs,
                                use_cuda=self.use_cuda,
                                gradient_clip_value=self.clip_grad_norm,
                                verbose=self.verbose,
                                batch_size=self.batch_size,
                                dataset=EmbeddingsSeqDataset,
                                dataset__model=model,
                                dataset__max_len=self.max_seq_len)
