"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import plotly.express as px
import pandas as pd
import numpy as np

test = pd.DataFrame([[5.1, 3.5, 0],
                     [4.9, 3.0, 0],
                     [7.0, 3.2, 1],
                     [6.4, 3.2, 1],
                     [5.9, 3.0, 2]],
                    columns=['length', 'width', 'species'])


def plot_list(x: list, y: list) -> None:
    """Lmao"""
    fig = px.scatter(x, y)
    fig.show()


def plot_scatter_df(data: pd.DataFrame, xax: str, yax: str) -> None:
    """Kill me"""
    fig = px.scatter(data, x=xax, y=yax)
    fig.show()
