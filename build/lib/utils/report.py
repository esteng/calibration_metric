import numpy as np
from typing import List, Dict
from metric import Metric

class Report:
    def __init__(self, metrics: Dict[str, Metric]):
        self.metrics = metrics

    def create_report(self, top_preds: np.array, is_correct: np.array):

        report = []
        for metric_name, metric in self.metrics.items():
            mse_val = metric(top_preds, is_correct)
            report.append(f"{metric_name}: {mse_val:.4f}")

        return "\n".join(report)


