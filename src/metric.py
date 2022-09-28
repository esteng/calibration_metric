from typing import Tuple
import pdb 

from collections import Counter
import numpy as np 
import scipy 
from scipy import stats
import pandas as pd 

from utils.warnings import check_size_warning

class Metric:
    def __init__(self, 
                name: str,
                n_bins: int = 20,
        ):
        self.name = name
        self.n_bins = n_bins

    def __call__(self, top_preds: np.array) -> float:
        raise NotImplementedError

    def bin_preds(self, 
                 top_probs: np.array, 
                 is_correct: np.array) -> Tuple[np.array, np.array, np.array]: 
        
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
    def __init__(self,
                n_bins: int = 20):
        super().__init__("MAE", n_bins)

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:

        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        check_size_warning(p_correct, p_model, self.name)

        mae = np.mean(np.abs(p_model - p_correct))
        return mae

class MeanErrorAbove(Metric):
    def __init__(self,
                n_bins: int = 20):
        super().__init__("Mean Error (overconfident)", n_bins)

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
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
    def __init__(self, 
                n_bins: int = 20):
        super().__init__("Mean Error (underconfident)", n_bins)

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
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