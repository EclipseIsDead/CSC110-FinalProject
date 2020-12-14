"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import ssl
from typing import List

TEST_SET = ["An eye-opening article. This further reinforces the need to switch to a more enviroment \
            friendly lifestyle. @EamonRyan thank you for sharing this!",
            "Bangladesh Confronts Climate Change - book goes open access http://upflow.co/l/Rc88",
            "If there’s a definition of insanity it’s Australia’s policies on climate change \
            being disrupted for a decade by wilfully ignorant and opportunistic politicians in \
            concert with cynical, deceitful and greedy fossil fuel lobbyists. They fiddle; our kids burn."]


def sentiment(sentences: List[str]) -> List[List[int]]:
    """
    Execute VADER analysis. Return a list of lists, where [negative, neutral, positive, compound]
    are the scores given.

    >>> sentiment(TEST_SET)
    [[0.0, 0.665, 0.335, 0.843], [0.192, 0.808, 0.0, -0.2263], [0.329, 0.671, 0.0, -0.9136]]
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('vader_lexicon')
    darth = SentimentIntensityAnalyzer()
    collector = []
    for sentence in sentences:
        ss = darth.polarity_scores(sentence)
        temp = []
        for k in ss.values():
            temp.append(k)
        collector.append(temp)
    return collector


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['nltk.sentiment.vader', 'nltk', 'ssl'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()
