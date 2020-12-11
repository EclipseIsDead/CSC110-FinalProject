"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import nltk
from nltk.corpus import treebank

# tweet id: 1028954283822260225
test = "An eye-opening article. This further reinforces the need to switch to a more enviroment \
        friendly lifestyle. @EamonRyan thank you for sharing this!"


def tree(sentence: str) -> None:
    """
    Display the parse tree of a sentence.
    Tokenize and tag a word using nltk library.
    Separates the sentence into tokens and then tags them, after this parse and display.
    :param sentence:
    :return:
    """
    tokens = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tags)
    t = treebank.parsed_sents('wsj_0001.mrg')[0]
    t.draw()
