# cython: profile=False

import collections


class VocabRecord(object):
    DEFAULT_PROPS = {'idx', 'count'}

    def __init__(self, idx, count=0.0, **other_props):
        self.idx = idx
        self.count = count
        vars(self).update(other_props)


class Vocabulary(object):
    RECORD_CLASS = VocabRecord

    def __init__(self, max_words_final=300000, min_count=5, truncate_at=10000000):
        self.max_words_final = max_words_final
        self.truncate_at = truncate_at
        self.min_count = min_count
        self.idx2word = []
        self.words = collections.defaultdict(int)

    def update(self, new_words):
        vocab = self.words
        for w in new_words:
            vocab[w] += 1
        if len(self.words) > self.truncate_at:
            self.truncate()

    def finalize(self):
        self.truncate()
        self._sort_words_by_freq()
        self._finalize_vocab_records()

    def _sort_words_by_freq(self):
        self.idx2word = list(self.words)
        self.idx2word.sort(key=lambda w: self.words[w],
                           reverse=True)

    def _finalize_vocab_records(self):
        old_vocab = self.words
        self.words = {w: self.RECORD_CLASS(i, count=old_vocab[w])
                      for i, w in enumerate(self.idx2word)}

    def _truncate_by_freq(self):
        words = self.words
        mc = self.min_count
        for w in list(words):
            if words[w] < mc:
                del words[w]

    def truncate(self):
        self._truncate_by_freq()

        if len(self.words) > self.max_words_final:
            self._sort_words_by_freq()

            for w in self.idx2word[self.max_words_final:]:
                del self.words[w]
            self.idx2word = self.idx2word[:self.max_words_final]

    def __contains__(self, word):
        return word in self.words

    def __getitem__(self, word):
        return self.words[word]

    def __len__(self):
        return len(self.words)


def build_vocab_from_dicts_default(vocab, data):
    for sample in data:
        vocab.update((sample['current'],))
    vocab.finalize()
