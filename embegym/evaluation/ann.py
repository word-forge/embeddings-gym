import annoy, itertools

from embegym.utils.io import default_save, default_load


class EmbeddingsIndex(object):
    def __init__(self, dimensions):
        self.word2idx = {}
        self.idx2word = []
        self.dimensions = dimensions
        self.idx = annoy.AnnoyIndex(dimensions)
        self.finalized = False

    def add_item(self, word, vec):
        assert not self.finalized, 'Could not add item to finalized embeddings index'
        idx = self.word2idx.get(word, None)
        if idx is None:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word.append(word)
        self.idx.add_item(idx, vec)

    def build(self, trees_number=100):
        self.finalized = True
        self.idx.build(trees_number)

    def save(self, out_file):
        default_save(self.word2idx, out_file)
        default_save(self.idx2word, out_file + '.idx2word')
        self.idx.save(out_file + '.annoy')

    def load(self, in_file):
        self.word2idx = default_load(in_file)
        self.idx2word = default_load(in_file + '.idx2word')
        self.idx.load(in_file + '.annoy')

    def __contains__(self, word):
        return word in self.word2idx

    def __getitem__(self, word):
        return self.idx.get_item_vector(self.word2idx[word])

    def get_nns_by_vector(self, vector, top_n=10, search_k=-1, include_distances=False):
        idx_res = self.idx.get_nns_by_vector(vector,
                                             top_n,
                                             search_k=search_k,
                                             include_distances=include_distances)
        if idx_res:
            if include_distances:
                return [(self.idx2word[i], dist) for i, dist in itertools.izip(*idx_res)]
            else:
                return [self.idx2word[i] for i in idx_res]
        else:
            return []

    def get_nns_by_item(self,
                        word,
                        top_n=10,
                        search_k=-1,
                        include_distances=False):
        idx_res = self.idx.get_nns_by_item(self.word2idx[word],
                                           top_n,
                                           search_k=search_k,
                                           include_distances=include_distances)
        if idx_res:
            if include_distances:
                return [(self.idx2word[i], dist) for i, dist in itertools.izip(*idx_res)]
            else:
                return [self.idx2word[i] for i in idx_res]
        else:
            return []


def build_annoy_idx(model, add_words=[], take_top_known_words=300000, n_trees=20):
    result = EmbeddingsIndex(model.vector_size())
    known_words = model.known_words()
    words_to_put_into_index = set(itertools.chain(known_words[:take_top_known_words],
                                                  add_words))
    for w in words_to_put_into_index:
        result.add_item(w, model.get_word_vector(w))

    result.build(n_trees)
    return result
