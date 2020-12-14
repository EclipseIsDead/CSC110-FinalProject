"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""

import csv
from typing import List, Dict
import datetime

KEYWORDS_LIST = ["trump", "breitbart", "cnn", "fakenews", "fake", "hurricane", "conspiracy",
                 "evidence", "study", 'deny',
                 'denier', 'science', 'energy', 'warm', 'weather',
                 'ocean', 'forest', 'fire', 'democrat', 'republican', 'hoax']

SENTIMENT_DATASET = 'datasets/twitter_sentiment_data.csv'
TWEETS_DATASET = 'datasets/tweets.csv'
TARGET_DATASET = 'datasets/clean_data.csv'


def convert_str_to_date(str_date: str) -> datetime.date:
    """
    This function is to be used when converting a csv to a pandas dataframe.
    Each date object is saved as a string, and must be converted back into a date object.

    >>> convert_str_to_date('2020-05-05')
    datetime.date(2020, 5, 5)

    >>> convert_str_to_date('2020-01-07')
    datetime.date(2020, 1, 7)
    """
    values = str_date.split('-')
    return datetime.date(int(values[0]), int(values[1]), int(values[2]))


def count_frequency(dataset: str, keywords: List[str]) -> Dict[str, int]:
    """
    Count_frequency takes a dataset and a list of keywords and counts the number of tweets
    that contains a given keyword. This function works for tweets.csv, not clean_data.csv
    >>> count_frequency(TWEETS_DATASET, ['trump'])
    {'trump': 155}
    """
    ret = {}
    for word in keywords:
        ret[word] = 0
    with open(dataset, 'r') as tweets:
        linereader = csv.reader(tweets, delimiter=',')
        for row in linereader:
            for word in keywords:
                if word in row[17]:
                    ret[word] += 1
    return ret


def convert_month_to_int(month: str) -> int:
    """
    This function takes a month in string form and returns the integer index of it
    >>> convert_month_to_int('Jan')
    1
    """
    ref_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    return ref_dict[month]


def convert_tweet_to_date(created_at: str) -> datetime.date:
    """
    This function takes a string in the format of the dataset and converts it to a date
    >>> convert_tweet_to_date('Mon Oct 31 18:21:25 +0000 2016')
    datetime.date(2016, 10, 31)
    """
    x = created_at.split()
    month = convert_month_to_int(x[1])
    day = int(x[2])
    year = int(x[5])
    return datetime.date(year, month, day)


def write_csv(target: str, header: List[str], tweets: list) -> None:
    """
    write_csv writes a list of tweets to the target csv file
    """
    with open(target, 'w') as new_dataset:
        writer = csv.writer(new_dataset)
        writer.writerow(header)
        for tweet in tweets:
            writer.writerow(tweet)


def clean_row(row: List[str], keywords: List[str], sentiment: Dict[int, int]) -> list:
    """
    clean_row takes a row from a dataset and retrieves the necessary information
    """
    tweet_date = convert_tweet_to_date(row[1])
    tweet_id = int(row[6])
    tweet_sentiment = sentiment[tweet_id]
    content = row[17]
    tweet_retweets = int(row[13])
    keyword_in_tweet = []
    for word in keywords:
        if word in content:
            keyword_in_tweet.append(1)
        else:
            keyword_in_tweet.append(0)
    return [tweet_sentiment, content, tweet_id, tweet_date, tweet_retweets] + keyword_in_tweet


def create_dataset(sentiment_data: str, dataset: str, target: str, keywords: List[str]) -> None:
    """
    Create dataset takes out current tweets.csv and creates a new csv file. In this new file
    each tweet is also notated with if it contains the given keyword. The date is also
    converted to datetime
    """
    columns = ['sentiment', 'content', 'tweet_id', 'date', 'retweets'] + keywords
    sentiment = dict()
    with open(sentiment_data, 'r') as sentiment_csv:
        reader = csv.reader(sentiment_csv, delimiter=',')
        x = 0
        for row in reader:
            if x != 0:
                tweet_id = int(row[2])
                tweet_sentiment = int(row[0])
                sentiment[tweet_id] = tweet_sentiment
            else:
                x = 1

    tweets = []
    with open(dataset, 'r') as data:
        reader = csv.reader(data, delimiter=',')
        x = 0
        for row in reader:
            if x != 0:
                tweets.append(clean_row(row, keywords, sentiment))
            else:
                x = 1
    write_csv(target, columns, tweets)


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['csv', 'datetime', 'python_ta.contracts'],
        'allowed-io': ['create_dataset', 'count_frequency', 'write_csv'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()
