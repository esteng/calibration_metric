from typing import Tuple
import pdb 

from collections import Counter
import numpy as np 
import scipy 
from scipy import stats
import pandas as pd 
import logging

from calibration_metric.utils.warnings import check_size_warning

logger = logging.getLogger(__name__)

class Metric:
    """
    Abstract class for metrics. 
    """
    def __init__(self, 
                name: str,
                n_bins: int = 20,
                weighted: bool = False,
                weight_key: str = "normalized_count",
        ):
        if weighted:
            name = f"Weighted {name}"
        self.name = name
        self.n_bins = n_bins
        self.weighted = weighted
        self.weight_key = weight_key

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        raise NotImplementedError

    def weight_by_count(self, p_correct: np.array, p_model: np.array, normalized_counts: np.array) -> np.array:
        abs_error = np.abs(p_correct - p_model) 
        weighted_mean_error = np.sum(abs_error * normalized_counts)
        return weighted_mean_error


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

        if any(np.isnan(values)):
            check_size_warning(top_probs, is_correct, self.name)
            logger.warn(f"NaN values in values from insufficient samples. Try decreasing n_bins or increasing the number of samples.")
            pre_values = np.array([x for x in values])
            pre_bins = np.array([x for x in bins])
            value_idxs = [i for i in range(len(values)) if not np.isnan(values[i])]
            nan_idxs = [i for i in range(len(values)) if np.isnan(values[i])]
            values = values[value_idxs]
            bins = bins[value_idxs]
            logger.warn(f"Reducing number of bins to {len(values)} by dropping NaN values at {nan_idxs} for bins {pre_bins[nan_idxs]}.")


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
        df['normalized_count'] = df['count'] / df['count'].sum()
        df['log_count'] = np.log(df['count']) 
        # NOTE: this is not the same as the log of the normalized count; it is intended to
        # discount high count bins.
        df['normalized_log_count'] = df['log_count'] / df['log_count'].sum()
        return df



class ECEMetric(Metric):
    """
    Computes expected calibration error (https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf)
    """
    def __init__(self,
                n_bins: int = 20,
                weighted: bool = True,
                return_df: bool = False,
                weight_key: str = "normalized_count"):
        super().__init__("ECE", n_bins, weighted, weight_key)
        self.return_df = return_df

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
        ece : float
            The expected calibration error 
        """
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        check_size_warning(p_correct, p_model, self.name)
        if self.weighted:
            norm_counts = df[self.weight_key].values
            ece = self.weight_by_count(p_correct, p_model, norm_counts)
        else:
            ece = np.mean(np.abs(p_model - p_correct))

        if self.return_df:
            return ece, df
        return ece 

class MCEMetric(Metric):
    """
    Computes maximum calibration error (https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf)
    """
    def __init__(self,
                n_bins: int = 20,
                weighted: bool = False,
                weight_key: str = "normalized_count"):
        super().__init__("MCE", n_bins, weighted, weight_key)

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
        mce : float
            The max calibration error 
        """
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        check_size_warning(p_correct, p_model, self.name)

        mce = np.max(np.abs(p_model - p_correct))
        return mce 


class MeanErrorAbove(Metric):
    """Computes the Mean Error on over-confident predictions."""
    def __init__(self,
                n_bins: int = 20,
                weighted: bool = False,
                weight_key: str = "normalized_count"):
        name = "Mean Error (overconfident)"
        super().__init__(name, n_bins, weighted, weight_key)

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

        if self.weighted:
            norm_counts = df[self.weight_key][p_model > p_correct].values
            return self.weight_by_count(over_p_correct, over_p_model, norm_counts)
        me = np.mean(over_p_model - over_p_correct)
        return me

class MeanErrorBelow(Metric):
    """Computes the Mean Error on under-confident predictions."""
    def __init__(self, 
                n_bins: int = 20,
                weighted: bool = False,
                weight_key: str = "normalized_count"):
        name = "Mean Error (underconfident)"
        super().__init__(name, n_bins, weighted, weight_key)

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
        if self.weighted:
            norm_counts = df[p_model < p_correct][self.weight_key].values
            return self.weight_by_count(under_p_correct, under_p_model, norm_counts)
        me = np.mean(under_p_correct - under_p_model)
        return me

class PearsonMetric(Metric):
    """
    Computes pearson correlation between prob under model and average correctness of bin. 
    """
    def __init__(self,
                n_bins: int = 20,
                weighted: bool = False,
                weight_key: str = "normalized_count"):
        super().__init__("Pearson", n_bins, weighted, weight_key)

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
        # if self.weighted:
            # norm_counts = df[self.weight_key].values
            # return self.weight_by_count(p_correct, p_model, norm_counts)

        # mae = np.mean(np.abs(p_model - p_correct))
        stat, p = stats.pearsonr(p_model, p_correct)
        return stat