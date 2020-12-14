"""
CSC110 Fall 2020 Final Project: Modeling Module

This module contains all functions related to creating a model of our data.
We create the model using XGBoost, which is an easy-to-use machine learning package.

University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
from typing import Tuple, List
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.graph_objects as go
import vader


def create_model(data: pd.DataFrame, labels: pd.DataFrame, max_depth: int, n_estimators: int) \
        -> Tuple[xgboost.XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create model will train an XGBoost classification model based on the given parameters.
    It returns a tuple with the model the training data, the training labels,
    the testing data, and the testing labels
    """
    train_data, test_data = train_test_split(data, random_state=1)
    train_labels, test_labels = train_test_split(labels, random_state=1)
    model = xgboost.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators)
    model.fit(train_data, train_labels)
    return (model, train_data, test_data, train_labels, test_labels)


def evaluate_model(test_labels: pd.Series, test_data: pd.DataFrame, model: xgboost.XGBClassifier) \
        -> float:
    """
    Returns a float which is the accuracy score of the model. Also creates graph
    of accuracy by predicted value.
    """
    predictions = model.predict(test_data)
    bins = [[], [], [], []]
    new_labels = test_labels.reset_index()
    for index, row in new_labels.iterrows():
        bins[row['sentiment'] + 1].append(predictions[index])
    neg_accuracy = get_accuracy(-1, bins[0]) * 100
    zero_accuracy = get_accuracy(0, bins[1]) * 100
    one_accuracy = get_accuracy(1, bins[2]) * 100
    two_accuracy = get_accuracy(2, bins[3]) * 100
    labels = [-1, 0, 1, 2]
    fig = go.Figure([go.Bar(x=labels, y=[
        neg_accuracy, zero_accuracy, one_accuracy, two_accuracy
    ])])
    fig.update_layout(
        title="Model Accuracy by Sentiment Score",
        xaxis_title="Sentiment Score",
        yaxis_title="Accuracy (%)",
    )
    fig.show()
    return accuracy_score(test_labels, predictions)


def get_accuracy(expected_label: int, predictions: List[int]) -> float:
    """
    This is used in the evaluate_model function. We return the percentage of
    items in the list that are equal to the expected_label.

    >>> get_accuracy(5, [5, 3, 2, 5])
    0.5
    >>> get_accuracy(1, [1, 3, 7, 10])
    0.25
    """
    length = len(predictions)
    matching = 0
    for i in predictions:
        if i == expected_label:
            matching += 1
    return matching / length


def run_example_model() -> None:
    """
    Example of using the create_model function and evaluation on dataset
    """
    clean_csv = pd.read_csv('datasets/clean_data.csv')
    sentiments = vader.sentiment(clean_csv['content'].tolist())
    clean_csv['vader'] = [x[3] for x in sentiments]
    data = clean_csv.drop(['sentiment', 'content', 'tweet_id', 'date'], axis=1)
    labels = clean_csv['sentiment']
    model, _, test_data, _, test_labels = create_model(data, labels, 6, 5)
    print('The accuracy of this model is: ', evaluate_model(test_labels, test_data, model), '%')


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['xgboost', 'sklearn.metrics', 'sklearn.model_selection',
                          'pandas', 'vader', 'plotly.graph_objects', 'python_ta.contracts'],
        'allowed-io': ['evaluate_model', 'create_model', 'run_example_model'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()
