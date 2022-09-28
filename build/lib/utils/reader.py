from typing import Tuple
import json 
import numpy as np 
class Reader:
    def __init__(self, file):
        self.file = file

    def read(self):
        raise NotImplementedError

class TopLogitFormatSequenceReader(Reader):

    def read(self) -> Tuple[np.array]:
        all_top_preds = []
        all_is_correct = []
        with open(self.file, 'r') as f:
            for line in f:
                line = json.loads(line)
                top_logits = np.array(line['top_logits'])
                top_logit_idxs = np.array(line['top_logit_idxs'])
                logit_at_label = np.array(line['logit_at_label'])
                labels = np.array(line['labels'])
                input_str = np.array(line['input_str'])
                is_batched = top_logits.shape[0] > 1

                if is_batched: 
                    # TODO: implement batched version
                    raise NotImplementedError(f"Currently batched outputs are not supported.\
                         Try generating outputs a single example at a time.")
                top_logits = top_logits[0]
                top_logit_idxs = top_logit_idxs[0]
                logit_at_label = logit_at_label[0]
                labels = labels[0]
                input_str = input_str[0]
                is_correct = top_logit_idxs == labels

                for timestep in range(top_logits.shape[0]):
                    all_top_preds.append(top_logits[timestep])
                    all_is_correct.append(is_correct[timestep])
                # to_yield = {"top_logits": top_logits,
                            # "top_logit_idxs": top_logit_idxs,
                            # "logit_at_label": logit_at_label,
                            # "labels": labels,
                            # "is_correct": is_correct,
                            # "input_str": input_str}
                # yield to_yield
        return (np.array(all_top_preds), np.array(all_is_correct)) 

