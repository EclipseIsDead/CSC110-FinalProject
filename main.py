"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import pandas as pd
import vader
import plotting


if __name__ == "__main__":
    print('Beginning Sentiment Analysis of Tweets...')
    print('Importing and Reading Dataset...')
    data = pd.read_csv('datasets/clean_data.csv')
    print('Conducting Frequency Analysis using Pandas...')
    sentiments = vader.sentiment(data['content'].tolist())
    data['vader'] = [x[3] for x in sentiments]
    data_date_and_freq = [data['date'], data['hoax'], data['hurricane'], data['fake']]
    headers = ['Date', 'hoax', 'hurricane', 'fake']
    date_and_freq = pd.concat(data_date_and_freq, axis=1, keys=headers)
    print('Plotting Frequency of Keywords towards Climate Change over Time...')
    date_and_freq.index = pd.to_datetime(date_and_freq.Date, format='%Y-%m-%d')
    date_and_freq = date_and_freq.groupby(pd.Grouper(freq='M')).sum()
    plotting.plot_proper(date_and_freq, 'Date', 'Frequency', 'tweet',
                         'Changes in Word Frequency Over Time')
    print('Conducting Sentiment Analysis over time using VADER...')
    data_date_and_vader = [data['date'], data['vader']]
    headers = ['Date', 'Sentiment']
    date_and_vader = pd.concat(data_date_and_vader, axis=1, keys=headers)
    print('Plotting Sentiment Analysis over time...')
    date_and_vader.index = pd.to_datetime(date_and_vader.Date, format='%Y-%m-%d')
    date_and_vader = date_and_vader.groupby(pd.Grouper(freq='M')).mean()
    date_and_vader['Date'] = date_and_vader.index
    plotting.plot_scatter_df(date_and_vader, 'Date', 'Sentiment', 'Sentiment Analysis over Time')
