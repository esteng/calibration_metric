import argparse
from utils.reader import TopLogitFormatSequenceReader
from utils.report import Report
from metric import MSEMetric, MeanErrorAbove, MeanErrorBelow

METRIC_LUT = {"mse": MSEMetric, "me_above": MeanErrorAbove, "me_below": MeanErrorBelow}
def create_reader(args):
    if args.reader == "top_logit_format_sequence":
        return TopLogitFormatSequenceReader(args.file)
    else:
        raise ValueError(f"Reader {args.reader} not supported")

def main(args):
    reader = create_reader(args) 
    top_preds, is_correct = reader.read()
    metric_names = args.metric_names.split(",")
    metrics = {metric_name: METRIC_LUT[metric_name]() for metric_name in metric_names}
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