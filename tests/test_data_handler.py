import pytest
from src.vlm.data.data_utils import DatasetHandler
from datasets import load_dataset


@pytest.fixture
def test_data_handler():
    data = load_dataset("priyank-m/chinese_text_recognition")

    result = DatasetHandler().merge_dataset([data["test"]])
