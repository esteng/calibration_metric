from typing import Tuple, List, Any
import json 
import numpy as np 
import pdb 

class Reader:
    """
    Abstract reader class, can be extended 
    to read different file formats
    """
    def __init__(self, file: str, ignore_tokens: List[Any] = None):
        self.file = file
        self.ignore_tokens = ignore_tokens

    def read(self):
        raise NotImplementedError

    def check_tokens(self, 
                    pred_tok: Any,
                    true_tok: Any): 
        """
        Basic check to compare tokens. 
        Cannot handle source copy
        """
        return pred_tok == true_tok

class TopLogitFormatSequenceReader(Reader):
    """
    Dataset reader for the HF output format.
    File format is jsonl, where each line is a 
    dict corresponding to a single input line,
    with the following keys:
    - top_logits: list of the top k logits for each timestep
    - top_logit_idxs: list of the top k logits for each timestep
    - labels: list of the label indices for each timestep
    """

    def read(self) -> Tuple[np.array]:
        """
        Read the file and extract the single top predicted index
        and the corresponding confidence score (logit).
        Compares each predicted index to the label index to determine
        whether the prediction was correct.
        Returns:
            top_preds: np.array of shape (num_examples, )
            is_correct: np.array of shape (num_examples, )
        """
        all_top_preds = []
        all_is_correct = []
        with open(self.file, 'r') as f:
            for line in f:
                line = json.loads(line)
                top_k_logits = np.array(line['top_logits'])
                top_k_logit_idxs = np.array(line['top_logit_idxs'])
                # get the top 1 logit and idx
                top_one_logit_local_idx = np.argmax(top_k_logits, axis=-1)
                seq_len = top_one_logit_local_idx.shape

                if len(seq_len) > 1: 
                    # TODO: implement batched version
                    raise NotImplementedError(f"Currently batched outputs are not supported.\
                         Try generating outputs a single example at a time.")

                seq_len = seq_len[0]

                # get the actual single top logit, not assuming they're sorted already 
                top_one_logit_local_idx = top_one_logit_local_idx.reshape((seq_len, 1))
                top_one_logit = np.take_along_axis(top_k_logits, top_one_logit_local_idx, axis=1)
                top_one_logit_idx = np.take_along_axis(top_k_logit_idxs, top_one_logit_local_idx, axis=1)
                labels = np.array(line['labels'])

                # currently only support single example per line 
                top_logits = top_one_logit.reshape(-1)
                top_logit_idxs = top_one_logit_idx.reshape(-1)

                is_correct = self.check_tokens(top_logit_idxs, labels) 

                for timestep in range(top_logits.shape[0]):
                    # ignore tokens if specified
                    # meant to be tokens like @ROOT@, BOS, EOS, etc.
                    if self.ignore_tokens is not None and labels[timestep] in self.ignore_tokens:
                        continue
                    all_top_preds.append(top_logits[timestep])
                    all_is_correct.append(is_correct[timestep])

        return (np.array(all_top_preds), np.array(all_is_correct))

class TopKTopLogitFormatSequenceReader(TopLogitFormatSequenceReader):
    def __init__(self, file: str, ignore_tokens: List[Any] = None, k: int = 2):
        super().__init__(file, ignore_tokens)
        self.k = k

    def check_tokens(self, 
                    pred_toks: List[Any],
                    true_tok: Any): 
        """
        Check whether the true token is contained in top k predicted tokens
        """
        return true_tok in pred_toks

    def read(self) -> Tuple[np.array]:
        """
        Read the file and extract the top k predicted indices
        and the corresponding confidence score (logit).
        Compares each predicted index to the label index to determine
        whether the prediction was correct.
        Returns:
            top_preds: np.array of shape (num_examples, )
            is_correct: np.array of shape (num_examples, )
        """
        all_top_preds = []
        all_is_correct = []
        with open(self.file, 'r') as f:
            for line in f:
                line = json.loads(line)
                top_k_logits = np.array(line['top_logits'])
                top_k_logit_idxs = np.array(line['top_logit_idxs'])
                # get the top k logit and idx
                top_k_logit_local_idxs = np.argsort(top_k_logits, axis=-1)[:, -self.k:]
                # top_one_logit_local_idx = np.argmax(top_k_logits, axis=-1)
                # seq_len = top_one_logit_local_idx.shape
                seq_len = top_k_logit_local_idxs.shape[0]


                # get the actual single top logit, not assuming they're sorted already 
                top_k_logit_local_idxs = top_k_logit_local_idxs.reshape((seq_len, -1))
                top_k_logits = np.take_along_axis(top_k_logits, top_k_logit_local_idxs, axis=1)
                top_k_logit_idxs = np.take_along_axis(top_k_logit_idxs, top_k_logit_local_idxs, axis=1)
                labels = np.array(line['labels'])

                for timestep in range(top_k_logits.shape[0]):
                    is_correct = self.check_tokens(top_k_logit_idxs[timestep], labels[timestep]) 
                    # ignore tokens if specified
                    # meant to be tokens like @ROOT@, BOS, EOS, etc.
                    if self.ignore_tokens is not None and labels[timestep] in self.ignore_tokens:
                        continue

                    # still use the single max logit as the confidence score 
                    all_top_preds.append(np.max(top_k_logits[timestep]))
                    all_is_correct.append(is_correct)

        return (np.array(all_top_preds), np.array(all_is_correct))

class MisoTopLogitFormatSequenceReader(TopLogitFormatSequenceReader):

    def check_tokens(self, 
                    pred_tok: str, 
                    tgt_tok: str, 
                    prev_tgts: List[str]) -> bool:
        """
        check if the predicted token is correct
        accounting for source and target copy 

        Parameters
        ----------
        pred_tok: str
            predicted token
        tgt_tok: str
            target token
        prev_tgts: List[str]
            previous target tokens for target copy 

        """
        if "SourceCopy" not in pred_tok and "TargetCopy" not in pred_tok:
            return pred_tok == tgt_tok
        elif "SourceCopy" in pred_tok:
            return pred_tok.split("_")[1] == tgt_tok
        else:
            try:
                # try target copy 
                tok_idx = int(pred_tok.split("_")[1])-1
                return prev_tgts[tok_idx] == tgt_tok
            except IndexError:
                # this should never happen 
                raise AssertionError

    def read(self) -> Tuple[np.array]:
        """
        Read the file and extract the single top predicted index
        and the corresponding confidence score (logit).
        Compares each predicted index to the label index to determine
        whether the prediction was correct.
        Returns:
            top_preds: np.array of shape (num_examples, )
            is_correct: np.array of shape (num_examples, )
        """
        all_top_preds = []
        all_is_correct = []
        with open(self.file, 'r') as f:
            for line in f:
                line = json.loads(line)
                top_k_logits = np.array(line['top_logits'])
                top_k_logit_idxs = np.array(line['top_logit_idxs'])
                # get the top 1 logit and idx
                top_one_logit_local_idx = np.argmax(top_k_logits, axis=-1)
                seq_len = top_one_logit_local_idx.shape

                if len(seq_len) > 1: 
                    # TODO: implement batched version
                    raise NotImplementedError(f"Currently batched outputs are not supported.\
                         Try generating outputs a single example at a time.")

                seq_len = seq_len[0]

                # get the actual single top logit, not assuming they're sorted already 
                top_one_logit_local_idx = top_one_logit_local_idx.reshape((seq_len, 1))
                top_one_logit = np.take_along_axis(top_k_logits, top_one_logit_local_idx, axis=1)
                top_one_logit_idx = np.take_along_axis(top_k_logit_idxs, top_one_logit_local_idx, axis=1)
                labels = np.array(line['labels'])

                # currently only support single example per line 
                top_logits = top_one_logit.reshape(-1)
                top_logit_idxs = top_one_logit_idx.reshape(-1)

                is_correct = []
                for idx in range(len(top_logits)):
                    # left context: everything up to current token
                    left_context = labels[0:idx]
                    tokens_are_equal = self.check_tokens(top_logit_idxs[idx], labels[idx], left_context)
                    is_correct.append(tokens_are_equal)

                # flatten  
                for timestep in range(top_logits.shape[0]):
                    # ignore tokens if specified
                    # meant to be tokens like @ROOT@, BOS, EOS, etc.
                    if self.ignore_tokens is not None and labels[timestep] in self.ignore_tokens:
                        continue
                    all_top_preds.append(top_logits[timestep])
                    all_is_correct.append(is_correct[timestep])

        return (np.array(all_top_preds), np.array(all_is_correct))