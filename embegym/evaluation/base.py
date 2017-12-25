import abc
import collections

import joblib
import pandas


class BaseEvaluator(object):
    @abc.abstractmethod
    def __call__(self, model):
        """
        Evaluate model and return scores
        :param model:
        :return: float or dict(str=float) - some numbers regarding quality of the model against this evaluation
        """
        pass


def _call_evaluation(eval_title, evaluator, model_title, model):
    return eval_title, model_title, evaluator(model)


class Evaluation(object):
    def __init__(self, evaluators, n_jobs=1):
        """
        :param evaluators: list of pairs (evaluator_name, evaluator obj or functor)
        :param n_jobs: int, number of jobs to pass to joblib
        """
        self._evaluators = evaluators
        self._n_jobs = n_jobs

    def __call__(self, *models, **named_models):
        models_to_evaluate = collections.OrderedDict()
        models_to_evaluate.update(('model_{}'.format(i), m)
                                  for i, m in enumerate(models))
        models_to_evaluate.update(named_models)

        flat_eval_results = joblib.Parallel(n_jobs=self._n_jobs)(
            joblib.delayed(_call_evaluation)(eval_title, evaluator, model_title, model)
            for eval_title, evaluator in self._evaluators
            for model_title, model in models_to_evaluate.items()
        )
        full_eval_results = collections.defaultdict(dict)
        for eval_title, model_title, cur_res in flat_eval_results:
            if isinstance(cur_res, collections.abc.Mapping):
                for val_title, val in cur_res.items():
                    full_eval_results[model_title][(eval_title, val_title)] = val
            else:
                full_eval_results[model_title][(eval_title, '')] = cur_res

        all_model_titles = list(models_to_evaluate.keys())
        result = pandas.DataFrame(data=[full_eval_results[model_title]
                                        for model_title in all_model_titles],
                                  index=all_model_titles)
        result.columns = pandas.MultiIndex.from_tuples(result.columns)
        return result
