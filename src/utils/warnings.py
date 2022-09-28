import warnings
import pdb 

def check_size_warning(p_model, p_correct, metric_name):
    if p_model.shape[0] == 0:
        warnings.warn(f"Metric {metric_name} has no active bins, returning -1", RuntimeWarning)
    if p_model.shape[0] < 3:
        warnings.warn(f"Metric {metric_name} has less than 3 active bins", RuntimeWarning)