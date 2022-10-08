import argparse
from calibration_metric.utils.reader import (TopLogitFormatSequenceReader, 
                                            MisoTopLogitFormatSequenceReader)
from calibration_metric.utils.report import Report
from calibration_metric.metric import (ECEMetric, 
                                        MeanErrorAbove, 
                                        MeanErrorBelow, 
                                        PearsonMetric,
                                        MCEMetric)
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

METRIC_LUT = {"ece": ECEMetric, 
             "me_above": MeanErrorAbove, 
             "me_below": MeanErrorBelow, 
             "pearson": PearsonMetric, 
             "max": MCEMetric}

def create_reader(args):
    """
    Create a dataset reader based on args and the file format 
    """
    if args.reader == "top_logit_format_sequence":
        return TopLogitFormatSequenceReader(args.logit_file, args.ignore_tokens)
    elif args.reader == "miso_logit_format_sequence": 
        return MisoTopLogitFormatSequenceReader(args.logit_file, args.ignore_tokens)
    else:
        raise ValueError(f"Reader {args.reader} not supported")

def main(args):
    def intify(x):
        try:
            return int(x)
        except ValueError:
            return x 

    if args.ignore_tokens is not None:
        ignore_tokens = args.ignore_tokens.split(",")
        args.ignore_tokens = [intify(token) for token in ignore_tokens ]

    logger.info("Creating logit file reader")
    reader = create_reader(args) 
    logger.info(f"Reading data from {args.logit_file}...")
    top_preds, is_correct = reader.read()
    logger.info(f"Finished reading {len(top_preds)} predictions")
    metric_names = args.metrics.split(",")
    logger.info(f"Creating {len(metric_names)} metrics...")
    metrics = {metric_name: METRIC_LUT[metric_name](args.n_bins, 
                                                    weighted=args.weighted,
                                                    weight_key=args.weight_key)
                for metric_name in metric_names}
    logger.info("Metrics:")
    report = Report(metrics)
    to_print = report.create_report(top_preds, is_correct)
    print(to_print)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logit_file", type=str, required=True)
    parser.add_argument("--reader", type=str, required=False, default="top_logit_format_sequence", choices = ["top_logit_format_sequence", "miso_logit_format_sequence"])
    parser.add_argument("--metrics", type=str, required=False, default="ece,me_above,me_below")
    parser.add_argument("--n_bins", type=int, default=20)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--weight_key", type=str, default="normalized_count", choices=["normalized_count", "normalized_log_count"])
    parser.add_argument("--ignore_tokens", type=str, default=None)

    args = parser.parse_args()
    main(args)