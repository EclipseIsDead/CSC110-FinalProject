"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import plotly.express as px
import numpy as np


def plot_scatter(x: list, y: list) -> None:
    """Kill me"""
    fig = px.scatter(x, y)
    fig.show()
