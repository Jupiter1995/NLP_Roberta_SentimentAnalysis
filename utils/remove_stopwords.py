import spacy
from spacy.tokenizer import *

from nltk.corpus import stopwords

from gensim.parsing.preprocessing import STOPWORDS as SW


class StopwordsRemove:

    def __init__(self, custom_stopwords: list[str] = None):
        self.nlp = spacy.load('en_core_web_md')
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.custom_stopwords = custom_stopwords

    def stop_words_collection(self):
        """
        :return: All the stopwords from three different sources - nltk, gensim, customized stopwords
        """
        custom_stopwords = self.custom_stopwords
        nltk_stopwords = stopwords.words('english')
        gensim_stopwords = SW

        # Combine all the stop words into one 
        ALL_STOP_WORDS = self.nlp.Defaults.stop_words.union(gensim_stopwords) \
            .union(nltk_stopwords) \
            .union(custom_stopwords)

        return ALL_STOP_WORDS

    def rm_stopwords(self, s, tokenizer_flag: bool = False) -> list[list[str]]:
        """
        :param s: raw corpus that still contains stopwords
        :param tokenizer_flag: (optional) default = False, when it is True, a predefined spacy tokenizer will be applied
        to the raw corpus before removing stopwords; when it is False，no tokenizer will be applied before removing
        stopwords.

        :return: corpus without stopwords
        """
        tokens = []
        ALL_STOP_WORDS = self.stop_words_collection()

        if tokenizer_flag == True:
            s = self.tokenizer.pipe(s, batch_size=500)
            # print(f"with tokenizer: {s}")

            for doc in s:
                doc_tokens = []
                for token in doc:
                    if token.text.lower() not in ALL_STOP_WORDS and len(token) > 2:
                        doc_tokens.append(token.text.lower())
                tokens.append(doc_tokens)

        else:
            for doc in s:
                doc_tokens = []
                for token in doc:
                    if (token.lower() not in ALL_STOP_WORDS) & (len(token) > 2):
                        doc_tokens.append(token.lower())
                tokens.append(doc_tokens)

        return tokens
