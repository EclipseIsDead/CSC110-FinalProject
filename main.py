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
    data = pd.read_csv('datasets/tweets.csv')
    # I can't believe this doesn't work lmao
    print('Conducting Sentiment Analysis using VADER...')
    sentiment = vader.sentiment(data['text'].tolist())
    compound = [x[3] for x in sentiment]
    date = data['created_at'].tolist()
    print('Plotting Sentiment towards Climate Change over Time')
    plotting.plot_list(date, compound)
