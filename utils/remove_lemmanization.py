import re
import pandas as pd

import spacy

from cleantext.clean import clean


class RemoveUrlEmojiLemmanization():

    def __init__(self, data: pd.Series):
        self.nlp = spacy.load('en_core_web_md')
        self.data = data

    def rm_username(self, text: str) -> str:
        """
        :param text: a string to be cleaned
        :return: a string excluding username(s)
        """
        return re.sub('@[^\s]+', '', text)

    def get_lemmas(self, text: str):
        """
        :param text: a string
        :return: a string after lemmatization
        """""
        lemmas = []

        doc = self.nlp(text)

        for token in doc:
            if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON'):
                lemmas.append(token.lemma_)

        return lemmas

    def preprocess_text(self, convertTostring: bool = False) -> list[list[str]]:
        """
        :param convertTostring: When it is True, the return result will be like [['w1..wn'],['w1...wn'], ...]
        when it is False, the return result will be list[list[str]] (e.g., [['w1', ..., 'wn'],['w1', .., 'wn']...])
        :return: depends on convertTostring param, list[list[str]]
        """

        # Remove username, url and emoji from the data
        data = self.data.apply(clean, no_emoji=True, no_urls=True, lower=True, replace_with_url="") \
            .apply(self.rm_username)

        # Lemmanization
        data = data.apply(self.get_lemmas)

        if convertTostring:
            data = [' '.join(map(str, l)) for l in data]

        return data
