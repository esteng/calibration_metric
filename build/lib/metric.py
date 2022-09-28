from typing import Tuple
from functools import cache 

from collections import Counter
import numpy as np 
import scipy 
from scipy import stats
import pandas as pd 

class Metric:
    def __init__(self, 
                name: str,
                n_bins: int = 20,
        ):
        self.name = name
        self.n_bins = n_bins

    def __call__(self, top_preds: np.array) -> float:
        raise NotImplementedError

    @cache
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

    @cache
    def bins_to_df(self, 
        values: np.array,
        bin_edges: np.array,
        bin_number: np.array,
        ) -> pd.DataFrame:
        # create LUT for bin number to number of items in that bin 
        bin_lookup = Counter(bin_number)
        # instantiate df 
        df = pd.DataFrame(columns=["prob_model", "prob_correct", "count"])
        # populate df
        for i, (val, edge, bin_num) in enumerate(zip(values, bin_edges, bin_number)):
            df = df.append({"prob_model": edge, 
                            "prob_correct": val, 
                            "count": bin_lookup[i+1]}, 
                            ignore_index=True)
        return df

class MSEMetric(Metric):
    def __init__(self):
        super().__init__("MSE")

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        pred_ys = df["prob_model"].values
        true_ys = df["prob_correct"].values
        mse = np.mean((pred_ys - true_ys)**2)
        return mse

class MeanErrorAbove(Metric):
    def __init__(self):
        super().__init__("Mean Error (overconfident)")

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        pred_ys = df["prob_model"].values
        true_ys = df["prob_correct"].values
        overconfident_pred_ys = pred_ys[pred_ys > true_ys]
        overconfident_true_ys = true_ys[pred_ys > true_ys]

        me = np.mean(overconfident_pred_ys - overconfident_true_ys)
        return me

class MeanErrorBelow(Metric):
    def __init__(self):
        super().__init__("Mean Error (underconfident)")

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        pred_ys = df["prob_model"].values
        true_ys = df["prob_correct"].values
        underconfident_pred_ys = pred_ys[true_ys > pred_ys]
        underconfident_true_ys = true_ys[true_ys > pred_ys]

        me = np.mean(underconfident_true_ys - underconfident_pred_ys)
        return me