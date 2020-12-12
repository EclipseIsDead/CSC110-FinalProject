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


def vader_test(sentences: list) -> None:
    """
    This should print sentiment analysis for each sentence in sentences.
    *elaborate*
    :param sentences:
    :return:
    """
    darth = SentimentIntensityAnalyzer()
    for sentence in sentences:
        print(sentence)
        ss = darth.polarity_scores(sentence)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]), end='')
        print()
