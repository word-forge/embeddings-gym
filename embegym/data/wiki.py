import bz2
from gensim.corpora.wikicorpus import filter_wiki, extract_pages
from .base import BaseFileCorpus
from .tokenization import DEFAULT_TOKENIZER, DEFAULT_SENT_TOKENIZER


WIKI_DEFAULT_NAMESPACES_TO_FILTER = ('0',)


def fetch_wiki_texts(in_file,
                     namespaces_to_filter=WIKI_DEFAULT_NAMESPACES_TO_FILTER,
                     min_text_length=200):
    return ((title, clean_text, page_id)
            for title, text, page_id in extract_pages(bz2.BZ2File(in_file), namespaces_to_filter)
            for clean_text in (filter_wiki(text),)
            if clean_text.strip() >= min_text_length)


class Wikipedia(BaseFileCorpus):
    def __init__(self, filename, tokenizer=DEFAULT_TOKENIZER, **fetch_wiki_texts_kwargs):
        super(Wikipedia, self).__init__(filename)
        self.tokenizer = tokenizer
        self.fetch_wiki_texts_kwargs = fetch_wiki_texts_kwargs

    def _generate_samples(self):
        for _, txt, _ in fetch_wiki_texts(self.filename,
                                          **self.fetch_wiki_texts_kwargs):
            for sample in self.tokenizer(txt):
                yield sample


class WikipediaSentence(Wikipedia):
    def __init__(self, filename, tokenizer=DEFAULT_SENT_TOKENIZER, **kwargs):
        super(WikipediaSentence, self).__init__(filename, tokenizer=tokenizer, **kwargs)
