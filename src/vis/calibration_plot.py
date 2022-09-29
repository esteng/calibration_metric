import pandas as pd 
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

def plot_df(df: pd.DataFrame, 
            use_log_count: bool = True, 
            figsize: Tuple[int, int] = (5, 5)) -> plt.Figure:
    """
    Plot the binned probabilies and correct labels against the x=y line 
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns "prob_model", "prob_correct", "count"
    use_log_count : bool, optional
        Whether to use log of count for the size of the points, by default True
    figsize : Tuple[int, int], optional
        Size of the figure, by default (5, 5)

    Returns
    -------
    fig : plt.Figure
    """
    count_key = "count"
    if use_log_count:
        df["log_count"] = np.log(df["count"])
        count_key = "log_count"
    fig, ax = plt.subplots(figsize=figsize)

    # plot bins 
    sns.scatterplot(data=df, x = "prob_model", y="prob_correct", size=count_key, ax=ax, legend='brief')
    # plot y=x line
    xs_line = np.linspace(0,1,2)
    ys_line = xs_line
    sns.lineplot(x = xs_line, y=ys_line, ax=ax, color='black')

    return fig