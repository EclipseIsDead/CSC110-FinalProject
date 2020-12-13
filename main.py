"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import pandas as pd
import vader
import plotting
from datetime import date


if __name__ == "__main__":
    print('Beginning Sentiment Analysis of Tweets...')
    print('Importing and Reading Dataset...')
    data = pd.read_csv('datasets/clean_data.csv')
    print('Conducting Sentiment Analysis using VADER...')
    sentiments = vader.sentiment(data['content'].tolist())
    data['vader'] = [x[3] for x in sentiments]
    data_date_and_vader = [data['date'], data['vader']]
    headers = ['date', 'vader']
    date_and_vader = pd.concat(data_date_and_vader, axis=1, keys=headers)
    date_and_vader.set_index('date')
    print('Plotting Sentiment towards Climate Change over Time...')
    date_and_vader.index = pd.to_datetime(date_and_vader.date, format='%Y-%m-%d')
    date_and_vader.groupby(pd.Grouper(freq='M')).mean()
    plotting.plot_proper(date_and_vader, 'date', 'vader', 'sentiment', 'Sentiment Over Time')
