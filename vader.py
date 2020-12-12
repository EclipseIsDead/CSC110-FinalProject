"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')

test_set = ["An eye-opening article. This further reinforces the need to switch to a more enviroment \
            friendly lifestyle. @EamonRyan thank you for sharing this!",
            "Bangladesh Confronts Climate Change - book goes open access http://upflow.co/l/Rc88",
            "If there’s a definition of insanity it’s Australia’s policies on climate change \
            being disrupted for a decade by wilfully ignorant and opportunistic politicians in \
            concert with cynical, deceitful and greedy fossil fuel lobbyists. They fiddle; our kids burn."]


def main(sentences: list) -> list:
    """
    Execute VADER analysis. Return a list of lists, where [negative, neutral, positive, compound]
     are the scores given. If this were to be run on test_set, the result should be
     [[0.0, 0.665, 0.335, 0.843], [0.192, 0.808, 0.0, -0.2263], [0.329, 0.671, 0.0, -0.9136]]
    :param sentences:
    :return:
    """
    darth = SentimentIntensityAnalyzer()
    collector = []
    for sentence in sentences:
        ss = darth.polarity_scores(sentence)
        temp = []
        for k in ss.values():
            temp.append(k)
        collector.append(temp)
    return collector
