import re 
import json 
import torch 
import numpy as np 
from transformers import Seq2SeqTrainer, AutoTokenizer


class LogitGetter:
    """
    Take a huggingface model and return data in format that can be read by 
    calibration_metric.reader.TopLogitFormatSequenceReader
    """
    def __init__(self, 
                trainer: Seq2SeqTrainer, 
                tokenizer: AutoTokenizer, 
                top_k: int):
        self.trainer = trainer
        self.tokenizer = tokenizer 
        self.top_k = top_k 

    def __call__(self):
        to_ret = []
        inputs = {k: v.to(self.trainer.args.device) for k, v in inputs.items()}
        with torch.no_grad():
            # run through trainer to get token probs 
            outputs = self.trainer.model(**inputs)
            logits = outputs.logits
            logits = torch.exp(torch.log_softmax(logits, dim=-1))
            logits = logits.detach().cpu().numpy()
            # get the top-k so we don't have to store all logits 
            logits_top_k_idxs = np.argsort(logits, axis=-1)[:, :, -self.top_k:]
            logits_top_k = np.take_along_axis(logits, logits_top_k_idxs, axis=-1)
            batch_size = logits.shape[0]
            logits_top_k_idxs = logits_top_k_idxs.tolist()
            logits_top_k = logits_top_k.tolist()

            # get logits at label idxs 
            # handle pad tokens
            unsqueezed_labels = inputs['labels'].unsqueeze(-1)
            labels_to_gather = unsqueezed_labels.clone()
            labels_to_gather[unsqueezed_labels == -100] = 0
            logit_at_label = outputs.logits.gather(2, labels_to_gather)
            logit_at_label[unsqueezed_labels == -100] = -100
            
            logit_at_label = logit_at_label.squeeze(-1)
            logit_at_label = logit_at_label.detach().cpu().numpy().tolist()
            labels = inputs['labels'].detach().cpu().numpy().tolist()
            inputs = inputs['input_ids'].detach().cpu().numpy().tolist()
            input_str = [self.tokenizer.decode(x) for x in inputs] 

        for batch_idx in range(batch_size): 
            # trim off padding
            instance_logit_at_label = logit_at_label[batch_idx]
            instance_logit_at_label = [x for x in instance_logit_at_label if x != -100]
            instance_labels = labels[batch_idx]
            instance_labels = [x for x in instance_labels if x != -100]
            instance_input_str = input_str[batch_idx]
            instance_input_str = re.sub("<pad>", "", instance_input_str)
            instance_top_logits = logits_top_k[batch_idx][0:len(instance_labels)]
            instance_top_logit_idxs = logits_top_k_idxs[batch_idx][0:len(instance_labels)]

            to_append = {"top_logits": instance_top_logits,
                        "top_logit_idxs": instance_top_logit_idxs,
                        "logit_at_label": instance_logit_at_label,
                        "labels": instance_labels,
                        "input_str": instance_input_str}
            to_ret.append(json.dumps(to_append))
        return to_ret 