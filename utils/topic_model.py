import re
import matplotlib.pyplot as plt
import pandas as pd

from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.test.utils import datapath


class TopicModel:

    def __init__(self, text: list[list[str]] = None, min_doc: int = 10, max_doc_frac: float = 0.85, *kwarg):
        """
        :param isPretrained: if it is True, a pretrained will be used for topic mining/modeling. Otherwise,
        a new model will be trained based on the input corpus.
        :param text: input corpus for training new model or retraining the pretrained model
        :param min_doc: minimum number of documents containing a word before excluding the word from consideration
        :param max_doc_frac: maximum fraction of the total number of documents containing a word before excluding the
        word from consideration
        """
        try:
            self.no_below = min_doc
            self.no_above = max_doc_frac
            
            self.id2word = Dictionary(text)
            self.id2word.filter_extremes(no_below=self.no_below, no_above=self.no_above)

            self.corpus = [self.id2word.doc2bow(doc) for doc in text]

        except:
            print('Please input a training text doc.')

    def fit(self, num_topics: int = 5, workers: int = 3, passes: int = 5):
        self.lda = LdaMulticore( 
            corpus=self.corpus, 
            num_topics=num_topics,
            id2word=self.id2word, 
            workers=workers,
            passes=passes
        )

    def transform(self) -> list[list[str]]:
        word_dist = [re.findall(r'"([^"]*)"', t[1]) for t in self.lda.print_topics()]
        topic_words = [[f"topic {t[0]}"] + w for t, w in zip(self.lda.print_topics(), word_dist)]
        return topic_words
    
    def fit_transform(self, num_topics: int = 5, workers: int = 3, passes: int = 5):
        self.fit(
            num_topics=num_topics,
            workers=workers,
            passes=passes
        )

        return self.transform()

    def save_trained_lda(self, save_path: str = None):
        try:
            self.lda.save(save_path)
        except:
            print('Please check the save_path parameter.')
    
    def topic_word_dist(self, top_n: int = 5, print_formatted: bool = False) -> list[list[str]]:
        """
        :param top_n: the number of top words for each topic to show
        :param print_formatted: if it is True, a formatted top_n words for each topic will be printed
        :return: list of all the topics and their associated words
        """
        word_dist = [re.findall(r'"([^"]*)"',t[1]) for t in self.lda.print_topics()]
        topic_words = [[f"topic {t[0]}"] + w[:top_n] for t, w in zip(self.lda.print_topics(), word_dist)]

        if print_formatted:   # Print word with weight and put each word in a new line for better reading experience
            res = []
            for t in self.lda.print_topics():
                string = t[1]

                # Remove parentheses and quotes
                string = string.replace('(','').replace(')','').replace('"','')

                # Split on commas and plus sign
                elements = string.split(' + ')

                # Split each element on asterisk and space characters and create list of lists
                result = [[float(e.split('*')[0]), e.split('*')[1].strip()] for e in elements]

                res.append((t[0], result))

            top_words = {}
            for words in res:
                for comb in words[1]:
                    num, word = comb[0], comb[1]
                    if word in top_words: continue
                    top_words[word] = num
                    break

            # plot the top words based on the probabilities
            names = list(top_words.keys())
            values = list(top_words.values())

            plt.bar(range(len(top_words)), values, tick_label=names)
            plt.xticks(rotation=90)
            plt.xlabel("Top 10 Words")
            plt.ylabel("Probability")
            plt.show()

        return topic_words
