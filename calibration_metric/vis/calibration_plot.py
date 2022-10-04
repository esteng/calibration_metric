import pandas as pd 
from typing import Tuple, Any, List
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 

from calibration_metric.metric import MAEMetric
from calibration_metric.utils.reader import Reader, TopLogitFormatSequenceReader

def get_df_from_file(file_path: str, 
                     n_bins: int=20,
                     reader_cls: Reader = TopLogitFormatSequenceReader,
                     ignore_tokens: List[Any] = None) -> pd.DataFrame:
    """
    Get the dataframe from the file path. 

    Parameters
    ----------
    file_path : str
        Path to the file

    Returns
    -------
    df : pd.DataFrame
    """
    metric = MAEMetric(n_bins)
    reader = reader_cls(file_path, ignore_tokens=ignore_tokens)
    top_preds, is_correct = reader.read()
    print(f"Number of examples: {len(is_correct)}")
    print(f"correct_sum: {np.sum(is_correct)}")
    values, bins, bin_number = metric.bin_preds(top_preds, is_correct)
    print(values)
    print(bins)
    df = metric.bins_to_df(values, bins, bin_number)
    return df

def plot_df(df: pd.DataFrame, 
            use_log_count: bool = True, 
            figsize: Tuple[int, int] = (5, 5),
            ax: Any  = None,
            title: str = None,
            show_legend: bool = True,
            xlabel: str = "Avg. Correct",
            ylabel: str = "Model Prob.") -> plt.Figure:
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

    if ax is None: 
        fig, ax = plt.subplots(figsize=figsize)

    # plot bins 
    sns.scatterplot(data=df, x = "prob_correct", y="prob_model", size=count_key, ax=ax, legend='brief')
    # plot y=x line
    xs_line = np.linspace(0,1,2)
    ys_line = xs_line
    sns.lineplot(x = xs_line, y=ys_line, ax=ax, color='black')
    sns.despine()
    if title is not None:
        ax.set_title(title)

    if not show_legend:
        ax.get_legend().remove()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # if ax is not None:
        # return fig