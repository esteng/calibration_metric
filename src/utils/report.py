import numpy as np
from typing import List, Dict
from metric import Metric

class Report:
    """Wrapper class to run and print metrics"""
    def __init__(self, metrics: Dict[str, Metric]) -> None:
        """
        Parameters
        ----------
        metrics : Dict[str, Metric]
            Dictionary of metric names to metric objects.
        """
        self.metrics = metrics

    def create_report(self, 
                    top_preds: np.array, 
                    is_correct: np.array) -> str:
        """
        Run all metrics and create a report.
        
        Parameters
        ----------
        top_preds : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array   
            Array of whether each timestep is correct for each timestep across all examples.
        
        Returns
        -------
        report : str
            String containing the report.
        """
        report = []
        for metric_name, metric in self.metrics.items():
            mse_val = metric(top_preds, is_correct)
            report.append(f"{metric_name}: {mse_val:.4f}")

        return "\n".join(report)


