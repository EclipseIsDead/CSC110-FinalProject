"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""

import csv
from typing import List, Dict
from datetime import date

keywords = ["trump", "breitbart", "cnn", "fakenews", "fake", "hurricane", "conspiracy", "evidence", "study", 'deny',
            'denier', 'science', 'energy', 'warm', 'weather',
            'ocean', 'forest', 'fire', 'democrat', 'republican', 'hoax']

sentiment = 'datasets/twitter_sentiment_data.csv'
dataset = 'datasets/tweets.csv'
target = 'datasets/clean_data.csv'


def count_frequency(dataset: str, keywords: List[str]) -> Dict[str, int]:
    """
    Count_frequency takes a dataset and a list of keywords and counts the number of tweets
    that contains a given keyword. This function works for tweets.csv, not clean_data.csv
    >>> count_frequency(dataset, ['trump'])
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
    dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
            'Sep': 9, 'Oct': 10,
            'Nov': 11, 'Dec': 12}
    return dict[month]


def convert_to_datetime(created_at: str) -> date:
    """
    This function takes a string in the format of the dataset and converts it to a date
    >>> convert_to_datetime('Mon Oct 31 18:21:25 +0000 2016')
    datetime.date(2016, 10, 31)
    """
    x = created_at.split()
    month = convert_month_to_int(x[1])
    day = int(x[2])
    year = int(x[5])
    return date(year, month, day)


def create_dataset(sentiment_data: str, dataset: str, target: str, keywords: List[str]) -> None:
    """
    Create dataset takes out current tweets.csv and creates a new csv file. In this new file, each tweet
    is also notated with if it contains the given keyword. The date is also converted to datetime
    """
    columns = ['sentiment', 'content', 'tweet_id', 'date', 'retweets'] + keywords
    tweets = []
    x = 0
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
                tweet_date = convert_to_datetime(row[1])
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
                tweets.append([tweet_sentiment, content, tweet_id, tweet_date, tweet_retweets] + keyword_in_tweet)
            else:
                x = 1

    with open(target, 'w') as new_dataset:
        writer = csv.writer(new_dataset)
        writer.writerow(columns)
        for tweet in tweets:
            writer.writerow(tweet)
