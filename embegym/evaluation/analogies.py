import itertools
import collections
import smart_open

from .base import BaseEvaluator
from .ann import build_annoy_idx
from embegym.utils.loader import try_get_resource


class AnalogiesBase(BaseEvaluator):
    def __init__(self, gold_data, rel_groups={}, depths=[1, 5], calc_coverage=False):
        """
        :param gold_data: dict, { class : list of analogy questions }, each question is a 4-tuple of words
        :param rel_groups: dict, { superclass : set of class titles }, a way to group syntactic-semantic-morphological relations together
        """
        self._gold_data = gold_data
        self._rel_groups = rel_groups
        self._depths = depths
        self._calc_coverage = calc_coverage
        assert len(self._depths) > 0

        self._max_depth = max(self._depths)

    def __call__(self, model):
        valid_questions_by_class = {cls: [question
                                          for question in question_lst
                                          if all(w in model for w in question)]
                                    for cls, question_lst in self._gold_data.items()}
        unique_words = {w
                        for qlst in valid_questions_by_class.values()
                        for query in qlst
                        for w in query}
        word2vector = {w: model.get_word_vector(w) for w in unique_words}

        embeddings_idx = build_annoy_idx(model, add_words=unique_words)

        pos_count_by_depth = {d: collections.defaultdict(float)
                              for d in self._depths}
        for cls, question_lst in valid_questions_by_class.items():
            for pair1a, pair1b, pair2a, pair2b in question_lst:
                query = word2vector[pair1a] + word2vector[pair2a] - word2vector[pair1b]
                # found_words_with_sims = model.get_most_similar(query, k=self._max_depth)
                all_found_words = embeddings_idx.get_nns_by_vector(query,
                                                                   top_n=self._max_depth)
                for depth in self._depths:
                    # found_words = {w for w, _ in found_words_with_sims[:depth]}
                    found_words = all_found_words[:depth]
                    if pair2b in found_words:
                        pos_count_by_depth[depth][cls] += 1

        result = {}
        if self._calc_coverage:
            result.update(('{} coverage'.format(cls), len(qlst) / len(self._gold_data[cls]))
                          for cls, qlst in valid_questions_by_class.items())
        result.update(('{} @{}'.format(cls, depth), pos_cnt / len(valid_questions_by_class[cls]))
                      for depth, depth_res in pos_count_by_depth.items()
                      for cls, pos_cnt in depth_res.items())

        for depth, depth_res in pos_count_by_depth.items():
            result['micro-mean @{}'.format(depth)] = \
                (sum(depth_res[cls] for cls in valid_questions_by_class.keys())
                 / sum(len(qlst) for qlst in valid_questions_by_class.values()))

        for super_cls, cls_lst in self._rel_groups.items():
            for depth, depth_res in pos_count_by_depth.items():
                result['{} @{}'.format(super_cls, depth)] = \
                    (sum(depth_res[cls] for cls in cls_lst)
                     / sum(len(valid_questions_by_class[cls]) for cls in cls_lst))

        return result


def read_google_analogies_file(fname, lowercase=True, generate_all_variants=False):
    result = {}
    cur_cls = None
    cur_questions = []
    with smart_open.smart_open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if line.startswith(':'):
                if cur_cls and cur_questions:
                    result[cur_cls] = cur_questions
                cur_cls = line[1:].strip()
                cur_questions = []
            else:
                pair1a, pair1b, pair2a, pair2b = line.split(' ')
                cur_questions.append((pair1a, pair1b, pair2a, pair2b))
                if generate_all_variants:
                    cur_questions.append((pair1b, pair1a, pair2b, pair2a))
                    cur_questions.append((pair2a, pair2b, pair1a, pair1b))
                    cur_questions.append((pair2b, pair2a, pair1b, pair1a))
    return result


_GOOGLE_ANALOGIES_RESOURCE_FILE = 'datasets/analogies/google-analogies.txt.gz'
_GOOGLE_ANALOGIES_SUPERCLASSES = {
    'semantic' : {
        'capital-common-countries',
        'capital-world',
        'currency',
        'city-in-state',
        'family'
    },
    'syntactic' : {
        'gram1-adjective-to-adverb',
        'gram2-opposite',
        'gram3-comparative',
        'gram4-superlative',
        'gram6-nationality-adjective',
    },
    'morphological' : {
        'gram5-present-participle',
        'gram7-past-tense',
        'gram8-plural',
    }
}

class GoogleAnalogies(AnalogiesBase):
    def __init__(self,
                 filename=_GOOGLE_ANALOGIES_RESOURCE_FILE,
                 rel_groups=_GOOGLE_ANALOGIES_SUPERCLASSES,
                 lowercase=True,
                 generate_all_variants=False,
                 depths=[1, 5]):
        data_filename = try_get_resource(filename, read=False)
        data = read_google_analogies_file(data_filename,
                                          lowercase=lowercase,
                                          generate_all_variants=generate_all_variants)
        super(GoogleAnalogies, self).__init__(data,
                                              rel_groups=rel_groups,
                                              depths=depths)
