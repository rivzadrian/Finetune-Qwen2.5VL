import pytest
from src.vlm.data.data_utils import DatasetHandler
from datasets import load_dataset, Dataset
from src.vlm.hyparams.data_args import DataArguments
from typing import List


def test_data_handler():
    data = load_dataset("priyank-m/chinese_text_recognition")
    result = DatasetHandler().merge_dataset([data["test"]], DataArguments)
    assert isinstance(result, Dataset)
