import abc


class BaseModel(object):
    @abc.abstractclassmethod
    def load(cls, path, *args, **kwargs):
        """
        Load model from file
        :param path: file to read
        :param args: optional arguments to loader
        :param kwargs: optional arguments to loader
        """
        pass

    @abc.abstractmethod
    def save(self, path):
        """
        Save model to file
        :param path:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_word_vector(self, word, *args, **kwargs):
        """
        Retrieve vector for word
        :param word: str, word to get vector for
        :param args: some additional context information
        :param kwargs: some additional context information
        :return: numpy.array - vector
        for the word, None otherwise
        """
        pass

    @abc.abstractmethod
    def __contains__(self, word):
        """
        Verify if this model can provide a meaningful vector for the word
        :param word: str, word to check
        :return: bool
        """
        pass

    @abc.abstractmethod
    def vector_size(self):
        """
        Get size of vectors
        :return: int
        """
        pass

    @abc.abstractmethod
    def export(self):
        """
        Retrieve essentials of embedding model: word->id mapping and embeddings matrix (or matrices)
        :return: tuple of 2 elements - list of str [ "i-th word" ] and list of numpy.arrays
        """
        pass

    @abc.abstractmethod
    def known_words(self):
        """
        Get a list of all known words
        :return: list of str
        """
        pass

    @abc.abstractmethod
    def get_most_similar(self, vector, k=10, *args, **kwargs):
        """
        Retrieve k most similar words for the given vector.
        :param vector: numpy.array - query vector to find similar for
        :param k: number of similar words to get
        :return: list of tuples (str, float) - [ ("j-th similar word", similarity) ]
        """
        pass


class TrainableModel(BaseModel):
    @abc.abstractmethod
    def reinitialize(self):
        """
        Re-initialize weights of the model (reset to zero or random)
        """
        pass

    @abc.abstractmethod
    def train(self, data):
        """
        Train embeddings using the given data
        :param data: any python sequence or generator that provides training samples
        """
        pass

import abc


class BaseModel(object):
    @abc.abstractclassmethod
    def load(cls, path, *args, **kwargs):
        """
        Load model from file
        :param path: file to read
        :param args: optional arguments to loader
        :param kwargs: optional arguments to loader
        """
        pass

    @abc.abstractmethod
    def save(self, path):
        """
        Save model to file
        :param path:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_word_vector(self, word, *args, **kwargs):
        """
        Retrieve vector for word
        :param word: str, word to get vector for
        :param args: some additional context information
        :param kwargs: some additional context information
        :return: numpy.array - vector
        for the word, None otherwise
        """
        pass

    @abc.abstractmethod
    def __contains__(self, word):
        """
        Verify if this model can provide a meaningful vector for the word
        :param word: str, word to check
        :return: bool
        """
        pass

    @abc.abstractmethod
    def vector_size(self):
        """
        Get size of vectors
        :return: int
        """
        pass

    @abc.abstractmethod
    def export(self):
        """
        Retrieve essentials of embedding model: word->id mapping and embeddings matrix (or matrices)
        :return: tuple of 2 elements - list of str [ "i-th word" ] and list of numpy.arrays
        """
        pass

    @abc.abstractmethod
    def known_words(self):
        """
        Get a list of all known words
        :return: list of str
        """
        pass

    @abc.abstractmethod
    def get_most_similar(self, vector, k=10, *args, **kwargs):
        """
        Retrieve k most similar words for the given vector.
        :param vector: numpy.array - query vector to find similar for
        :param k: number of similar words to get
        :return: list of tuples (str, float) - [ ("j-th similar word", similarity) ]
        """
        pass


class TrainableModel(BaseModel):
    @abc.abstractmethod
    def reinitialize(self):
        """
        Re-initialize weights of the model (reset to zero or random)
        """
        pass

    @abc.abstractmethod
    def train(self, data):
        """
        Train embeddings using the given data
        :param data: any python sequence or generator that provides training samples
        """
        pass


class MultiLingualModel(BaseModel):
    @abc.abstractclassmethod
    def load(cls, *paths, *languages, *args, **kwargs):
        """
        Load multiple models for different languages
        :param paths: files to read
        :param languages: languages of read files
        :param args: optional arguments to loader
        :param kwargs: optional arguments to loader
        """
        pass

    @abc.abstractmethod
    def get_word_vector(self, word, language, *args, **kwargs):
        """
        Retrieve vector for a word for a language
        :param word: str, word to get vector for
        :param language: str, language to get word for
        :param args: some additional context information
        :param kwargs: some additional context information
        :return: numpy.array - vector
        for the word, None otherwise
        """
        pass

    @abc.abstractmethod
    def __contains__(self, word, language):
        """
        Verify if this model can provide a meaningful vector for the word
        :param word: str, word to check
        :param language: str, language to check word for
        :return: bool
        """
        pass

    @abc.abstractmethod
    def vector_size(self):
        """
        Get size of vectors
        :return: int
        """
        pass

    @abc.abstractmethod
    def export(self):
        """
        Retrieve essentials of embedding model: word->id mapping and embeddings matrix (or matrices)
        :return: tuple of 2 elements - list of str [ "i-th word" ] and list of numpy.arrays
        """
        pass

    @abc.abstractmethod
    def known_words(self):
        """
        Get a list of all known words
        :return: list of str
        """
        pass

    @abc.abstractmethod
    def get_most_similar(self, vector, k=10, *args, **kwargs):
        """
        Retrieve k most similar words for the given vector in specified languages.
        :param vector: numpy.array - query vector to find similar for
        :param k: number of similar words to get
        :return: list of tuples (str, float) - [ ("j-th similar word", similarity) ]
        """
        pass

    @abc.abstractmethod
    def cross_lang_similarity(self, word1, word2, *args, **kwargs):
        """
        Retrieve similarity measure for two words.
        :param word1: str, first word
        :param word2: str, second word
        :return: float, similarity measure
        """
        pass
