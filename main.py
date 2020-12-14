"""
CSC110 Fall 2020 Final Project: Main Module

This module contains the synthesis of all of our work. When run, it does sentiment analysis
on our dataset and graphs important trends.

University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import pandas as pd
import vader
import plotting
import boosting


if __name__ == "__main__":
    print('Beginning Sentiment Analysis of Tweets...')
    print('Importing and Reading Dataset...')
    data = pd.read_csv('datasets/clean_data.csv')
    print('Conducting Frequency Analysis using Pandas...')
    sentiments = vader.sentiment(data['content'].tolist())
    data['vader'] = [x[3] for x in sentiments]
    headers = ['date', 'hoax', 'hurricane', 'fake']
    data_date_and_freq = [data[label] for label in headers]
    date_and_freq = pd.concat(data_date_and_freq, axis=1, keys=headers)
    print('Plotting Frequency of Keywords towards Climate Change over Time...')
    date_and_freq.index = pd.to_datetime(date_and_freq.date, format='%Y-%m-%d')
    date_and_freq = date_and_freq.groupby(pd.Grouper(freq='M')).sum()
    plotting.plot_proper(date_and_freq, 'Date (Month)', 'Keyword Occurrence', 'Keyword',
                         'Changes in Sentiment Over Time')
    print('Conducting Sentiment Analysis over time using VADER...')
    data_date_and_vader = [data['date'], data['vader']]
    headers = ['date', 'vader']
    date_and_vader = pd.concat(data_date_and_vader, axis=1, keys=headers)
    print('Plotting Sentiment Analysis over time...')
    date_and_vader.index = pd.to_datetime(date_and_vader.date, format='%Y-%m-%d')
    date_and_vader = date_and_vader.groupby(pd.Grouper(freq='M')).mean()
    date_and_vader['Time'] = date_and_vader.index
    print('The r_2 of vader sentiment over time is: ',
          plotting.plot_scatter_df(date_and_vader, 'Time', 'vader',
                                   'Vader Sentiment Analysis Over Time'))
    print('Training XGBoost Model')
    boosting.run_example_model()
