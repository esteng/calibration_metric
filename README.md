# calibration_metric: a package for measuring calibration in seq2seq tasks

This package provides functions and classes for measuring calibration. It's designed primarily for measuring and plotting calibration at the token-level for sequence-to-sequence tasks, especially semantic parsing. 
For more information, please see [the paper describing this library](https://arxiv.org/abs/2211.07443). If you use this library or find it helpful, please cite: 

```
@article{stengel2022calibrated,
  title={Calibrated Interpretation: Confidence Estimation in Semantic Parsing},
  author={Stengel-Eskin, Elias and Van Durme, Benjamin},
  journal={arXiv preprint arXiv:2211.07443},
  year={2022}
}
``` 

## Quickstart 
To install this package locally (assuming a `python=3.9` environment): 

```
git clone git@github.com:esteng/calibration_metric.git 
cd calibration_metric
pip install -e .
```

The install is designed to be lightweight; however, the examples in `examples` require some heavier dependencies (torch, datsets, etc.). 
Since the examples are not central to the functionality of the library, installing them is optional. 
If you would like to run the examples, install with the examples flag: 

```
pip install -e .[examples]
```

To test that your installation worked, try:

```
python -m calibration_metric.main --logit_file examples/logits/dev_medium.logits
```


## Metrics
This package supports standard metrics like expected calibration error (ECE) and max calibration error (MCE). 
It is easy to add your own metric by inheriting from the `calibration_metric.metric.Metric` class and defining a new `__call__` method. 
These metrics share a common structure: first, they take paired lists of binary accuracy labels (1 = correct, 0 = incorrect) and confidence scores ($\in [0,1]$)
and bin the accuracy labels by the confidence score. They then compute the mean accuracy for each bin.

A perfectly-calibrated model would have a perfect correlation between the confidence of a bin and the bin's average accuracy. 
The metrics implemented vary in how they assign error. The most common metric, ECE, assigns error based on the difference between accuracy and confidence for each bin, weighted by the size of the bin. 

The `Metric` class can be used to bin predictions and also to compute the metric. 
For example, given a list of binary accuracy variables `accs` and a list of confidences `probs`, computing ECE looks like 

```
ece_metric = ECEMetric(n_bins=20, weighted=True)
ece_score = ece_metric(accs, probs)
```

## Visualization
In addition to computing metrics, this library provides tools for visualizing calibration curves. 
The main tools for doing this are in `calibration_metric.vis.calibration_plot`. 
`get_df_from_file()` allows you to get a pandas Dataframe from a file in order to plot. Alternatively, given a list of accuracies `accs` and confidences `probs`, a dataframe can be obtained with the `Metric` class: 

```
ece_metric, df = ECEMetric(n_bins=20, weighted=True, return_df=True)(accs, probs)
```

This dataframe can be plotted using `plot_df()`: 

```
from matplotlib import pyplot as plt 
fig, ax = plt.subplot(1, 1, figsize=(5,5))
plot_df(df, ax = ax)
```

`plot_df` allows for ECE values to be overlaid onto the plot, as in Figure 1 of our paper. 

## Readers
This package also contains utils for reading in predicted tokens and token probabilities. 
The main reader is the `calibration_metric.utils.reader.TopLogitFormatSequenceReader`, which reads in sequences of logits and tokens.
The file format has each datapoint on a separate line. Each datapoint is a token-level dictionary. If $N$ is the number of tokens in the gold output sequence, $V$ is the vocabulary size, and $k$ is a small number of logits to store, then each dict has:
 - `top_logits`: $N \times k$: the top k logits as predicted by the model for each of the $N$ tokens 
 - `top_logit_idxs`: $N \times k$: the vocab indices of the top k logits for each of the $N$ tokens
 - `logit_at_label`: $N \times 1$: the logit at the gold label index
 - `labels`: $N \times 1$: the vocab indices of the gold labels 
 - `input_str`: The input string corresponding to the gold sequence (useful for analysis and debugging)

 If this seems complicated, don't worry: there is also a utility for generating these files.
 `calibration_metric.utils.hf_utils.LogitGetter` allows you to generate a file like this for any seq2seq model in HuggingFace. 
 An example of how to use `LogitGetter` can be found in `examples/hf_generate.py`.


 ## Example
 For an example of how to obtain metrics from a huggingface model, please see `examples/hf_generate.py`. 
 To make things easier, a script called `examples/get_logits.sh` is included that takes in a path to a model. 
 The models used here are models from [BenchClamp](https://github.com/microsoft/semantic_parsing_with_constrained_lm). 
 To make things easier, a pre-trained T5-small semantic parsing model for SMCalFlow can be downloaded [here](https://nlp.jhu.edu/semantic_parsing_calibration/t5_benchclamp_checkpoint_10000.tar.gz).
 Once you've downloaded the model, you can generate a logits file using `./examples/get_logits.sh <PATH_TO_MODEL>`. 
 This will save a logits file to the following directory: `<PATH_TO_MODEL>/output_logits`. 
