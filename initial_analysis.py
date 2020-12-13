"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import pandas as pd
import vader
import plotting
import plotly.express as px

data = pd.read_csv('datasets/twitter_sentiment_data.csv')
df = pd.read_csv('datasets/clean_data.csv')
data['message'] = data['message'].astype('str')
sentiments = vader.sentiment(data['message'].tolist())
data['vader'] = [x[3] for x in sentiments]

# counting = data['sentiment'].value_counts()

# data['difference'] = data['sentiment'] - data['vader']

# fig_per = px.line(data, x='tweetid', y='difference')
# fig_per.show()

# fig = px.line(data, x='tweetid', y='vader')
# fig.add_scatter(x=data['tweetid'], y=data['sentiment'], mode='lines')
# fig.show()
