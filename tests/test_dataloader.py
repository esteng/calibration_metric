import pytest
import sys
print(sys.path)

from utils.reader import TopLogitFormatSequenceReader

@pytest.fixture
def two_logit_file():
    return "tests/data/dev_two.logits"

def test_top_logit_format_sequence_reader(two_logit_file):
    reader = TopLogitFormatSequenceReader(two_logit_file)
    top_preds, is_correct = reader.read()
    assert(top_preds.shape == (171,))
    assert(is_correct.shape == top_preds.shape)



