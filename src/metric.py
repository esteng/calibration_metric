from typing import Tuple
import pdb 

from collections import Counter
import numpy as np 
import scipy 
from scipy import stats
import pandas as pd 

from utils.warnings import check_size_warning

class Metric:
    """
    Abstract class for metrics. 
    """
    def __init__(self, 
                name: str,
                n_bins: int = 20,
        ):
        self.name = name
        self.n_bins = n_bins

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        raise NotImplementedError

    def bin_preds(self, 
                 top_probs: np.array, 
                 is_correct: np.array) -> Tuple[np.array, np.array, np.array]: 
        """
        Bin predicted probabilities and correct binary labels into n_bins.
        Binning is done by predited probability, and each bin's value is 
        the average number of correct examples in that bin.

        Parameters
        ----------
        top_probs : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array
            Array of whether each timestep is correct for each timestep across all examples.

        Returns
        -------
        values : np.array
            (n_bins, 1), Array of the average number of correct examples in each bin.
        bin_edges : np.array
            (n_bins, 1), Array of the bin edges (probabilities)
        bin_number : np.array
            (n_examples, 1), Array of the bin number for each example.
        """ 
        try:
            assert(top_probs.shape[0] == is_correct.shape[0])
        except AssertionError:
            raise AssertionError(f"top_probs and is_correct must have the same length, got {top_probs.shape} and {is_correct.shape} respectively.")

        # bin predicted probs in n_bins bins 
        (values, 
        bins, 
        bin_number) = stats.binned_statistic(
            top_probs, 
            is_correct, 
            statistic='mean', 
            bins=self.n_bins
        )

        return (values, bins, bin_number)

    def bins_to_df(self, 
        values: np.array,
        bin_edges: np.array,
        bin_number: np.array,
        ) -> pd.DataFrame:
        """
        Convert the output of bin_preds to a pandas dataframe.
        DataFrame has following columns:
        - prob_model: the probability for the bin
        - prob_correct: the average number of correct examples in the bin
        - count: the number of examples in the bin
        """
        # create LUT for bin number to number of items in that bin 
        bin_lookup = Counter(bin_number)
        # instantiate df 
        # df = pd.DataFrame(columns=["prob_model", "prob_correct", "count"])
        # populate df
        df_data = []
        for i, (val, edge, bin_num) in enumerate(zip(values, bin_edges, bin_number)):
            df_data.append({"prob_model": edge, 
                            "prob_correct": val, 
                            "count": bin_lookup[i+1]})
        df = pd.DataFrame.from_dict(df_data)
        return df

class MAEMetric(Metric):
    """
    Computes mean absolute error against y=x line.
    """
    def __init__(self,
                n_bins: int = 20):
        super().__init__("MAE", n_bins)

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        """
        Parameters
        ----------
        top_preds : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array
            Array of whether each timestep is correct for each timestep across all examples.
        
        Returns
        -------
        mae : float
            Mean absolute error against y=x line.
        """
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        check_size_warning(p_correct, p_model, self.name)

        mae = np.mean(np.abs(p_model - p_correct))
        return mae

class MeanErrorAbove(Metric):
    """Computes the Mean Error on over-confident predictions."""
    def __init__(self,
                n_bins: int = 20):
        super().__init__("Mean Error (overconfident)", n_bins)

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        """
        Parameters
        ----------
        top_preds : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array
            Array of whether each timestep is correct for each timestep across all examples.
        
        Returns
        -------
        mae : float
            Mean error against y=x line for bins lying above the line.
        """
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        over_p_model  = p_model[p_model > p_correct]
        over_p_correct = p_correct[p_model > p_correct]
        check_size_warning(over_p_correct, over_p_model, self.name)

        if over_p_correct.shape[0] == 0:
            return -1.0
        me = np.mean(over_p_model - over_p_correct)
        return me

class MeanErrorBelow(Metric):
    """Computes the Mean Error on under-confident predictions."""
    def __init__(self, 
                n_bins: int = 20):
        super().__init__("Mean Error (underconfident)", n_bins)

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        """
        Parameters
        ----------
        top_preds : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array
            Array of whether each timestep is correct for each timestep across all examples.
        
        Returns
        -------
        mae : float
            Mean error against y=x line for bins lying below the line.
        """
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        under_p_model  = p_model[p_model < p_correct]
        under_p_correct = p_correct[p_model < p_correct]
        check_size_warning(under_p_correct, under_p_model, self.name)

        if under_p_correct.shape[0] == 0:
            return -1.0 
        me = np.mean(under_p_correct - under_p_model)
        return me