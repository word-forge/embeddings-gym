import functools
import itertools

import numpy
import pandas
import scipy

from embegym.utils.loader import try_get_resource
from .base import BaseEvaluator


def _normed_distance_to_similarity(f):
    @functools.wraps
    def _impl(a, b):
        return 1 - f(a, b)
    return _impl


class BaseWordSimCorrEvaluator(BaseEvaluator):
    def __init__(self,
                 gold_data,
                 distance=scipy.spatial.distance.cosine,
                 similarity=None,
                 correlation=scipy.stats.spearmanr):
        """
        :param gold_data: pandas.DataFrame with 2-level index (two words) and columns representing gold judgements
        :param distance: a function that takes two vectors and outputs a float
            (closer to 1 if vectors are farther from each other, closer to 0 otherwise)
        :param similarity: like 1-`distance`
        :param correlation: like `similarity`, but to compare lists of similarities
        """
        self._gold_data = gold_data

        assert self._gold_data.index.nlevels == 2
        self._unique_words = set(itertools.chain.from_iterable(self._gold_data.index.levels))
        self._numeric_columns = [col for col in self._unique_words.columns
                                 if numpy.issubdtype(self._unique_words[col].dtype, numpy.numeric)]

        assert (distance is None) ^ (similarity is None), 'Only one of distance and similarity must be specified'
        if distance:
            self._similarity = _normed_distance_to_similarity(distance)
        elif similarity:
            self._similarity = similarity

        self._correlation = correlation

    def __call__(self, model):
        word2vector = {model[w] for w in self._unique_words if w in model}
        known_word_pairs = [(a, b)
                            for a, b in self._gold_data.index
                            if a in word2vector and b in word2vector]
        pred_similarities = numpy.array([self._similarity(word2vector[a], word2vector[b])
                                         for a, b in known_word_pairs])
        gold_values = self._gold_data.loc[known_word_pairs]
        return { column : self._correlation(pred_similarities, gold_values[column])
                 for column in self._numeric_columns }


_SIM_LEX_RESOURCE_FILE = 'datasets/SimLex-999.txt.gz'


class SimLex999(BaseWordSimCorrEvaluator):
    def __init__(self):
        data_file = try_get_resource(_SIM_LEX_RESOURCE_FILE, read=False)
        data = pandas.read_csv(data_file,
                               sep='\t',
                               index_col=['word1', 'word2'])
        super(SimLex999, self).__init__(data)
