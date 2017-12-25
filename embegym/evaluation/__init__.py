from . import base, analogies, similarities, supervised
from embegym.utils.registry import Registry

_EVALUATORS = Registry()
_EVALUATORS.register('SimLex999', similarities.SimLex999())
_EVALUATORS.register('GoogleAnalogies', analogies.GoogleAnalogies())
_EVALUATORS.register('20ng_mean_logreg', supervised.text_cat.MeanEmbeddingLogreg20NG())
_EVALUATORS.register('20ng_lstm', supervised.text_cat.CnnLstm20NG())


def get_known_evaluations():
    return _EVALUATORS.names()


def register_evaluator(name, obj):
    _EVALUATORS.register(name, obj)


def make_evaluation(*evaluators, n_jobs=1):
    return base.Evaluation(_EVALUATORS.objects(evaluators),
                           n_jobs=n_jobs)
