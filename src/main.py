import argparse
from utils.reader import TopLogitFormatSequenceReader
from utils.report import Report
from metric import MAEMetric, MeanErrorAbove, MeanErrorBelow
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

METRIC_LUT = {"mse": MAEMetric, "me_above": MeanErrorAbove, "me_below": MeanErrorBelow}

def create_reader(args):
    """
    Create a dataset reader based on args and the file format 
    """
    if args.reader == "top_logit_format_sequence":
        return TopLogitFormatSequenceReader(args.logit_file)
    else:
        raise ValueError(f"Reader {args.reader} not supported")

def main(args):
    logger.info("Creating logit file reader")
    reader = create_reader(args) 
    logger.info(f"Reading data from {args.logit_file}...")
    top_preds, is_correct = reader.read()
    logger.info(f"Read {len(top_preds)} predictions")
    metric_names = args.metrics.split(",")
    metrics = {metric_name: METRIC_LUT[metric_name]() for metric_name in metric_names}
    logger.info("Metrics:")
    report = Report(metrics)
    to_print = report.create_report(top_preds, is_correct)
    print(to_print)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logit_file", type=str, required=True)
    parser.add_argument("--reader", type=str, required=False, default="top_logit_format_sequence")
    parser.add_argument("--metrics", type=str, required=False, default="mse,me_above,me_below")
    args = parser.parse_args()
    main(args)