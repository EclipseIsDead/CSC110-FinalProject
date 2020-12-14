"""
University of Toronto CSC110 Final Project: Sentiment Analysis of Climate Change Tweets
Siddarth Dagar, Bradley Mathi, Backer Jackson, Daniel Zhu
"""
import plotly.express as px
import pandas as pd


def plot_scatter_df(data: pd.DataFrame, xax: str, yax: str, title: str) -> float:
    """
    This function creates a scatter plot using a pandas dataframe.
    """
    pd.options.plotting.backend = "plotly"
    fig = px.scatter(data, x=xax, y=yax, title=title, trendline='ols')
    fig.update_layout(
        xaxis_title="Date (month)",
        yaxis_title='Average Vader Sentiment Score',
    )
    fig.show()
    return px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared


def plot_proper(df: pd.DataFrame, xax: str, yax: str, var: str, title: str) -> None:
    """
    This function creates a line chart using a pandas Dataframe.
    """
    pd.options.plotting.backend = "plotly"
    fig = df.plot(title=title, template="simple_white",
                  labels=dict(index=xax, value=yax, variable=var))
    fig.update_layout(
        xaxis_title=xax,
        yaxis_title=yax,
    )
    fig.show()


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['plotly.express', 'pandas', 'python_ta.contracts'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts

    python_ta.contracts.check_all_contracts()

    import doctest

    doctest.testmod()
