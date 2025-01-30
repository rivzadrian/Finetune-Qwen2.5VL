from datasets import (
    load_dataset,
    DatasetDict,
    concatenate_datasets,
    interleave_datasets,
    Dataset,
    IterableDataset,
)
from utils.logger import logger
from typing import List, Union, Any, Dict
from src.vlm.hyparams.data_args import DataArguments


class DatasetHandler:
    def convert2Dataset(
        self, all_datasets: List[Union[Dataset, IterableDataset, List[Dict], Dict]]
    ) -> List[Union[Dataset, IterableDataset]]:
        convert_datasets = []
        for sub_dataset in all_datasets:
            if isinstance(sub_dataset, Dict):
                convert_datasets.append(Dataset.from_dict(sub_dataset))
            elif isinstance(sub_dataset, List):
                convert_datasets.append(Dataset.from_list(sub_dataset))
            else:
                convert_datasets.append(sub_dataset)

        return convert_datasets

    def merge_dataset(
        self,
        all_datasets: List[Union[Dataset, IterableDataset, List[Any]]],
        data_args: DataArguments,
        shuffle: bool = False,
    ) -> Union[Dataset, IterableDataset]:
        """
        Merges multiple datasets to a unified dataset.
        """
        all_datasets = self.convert2Dataset(all_datasets)

        if len(all_datasets) == 1:
            return all_datasets[0]

        elif data_args.mix_strategy == "concat":
            if data_args.streaming:
                logger.warning(
                    "The samples between different datasets will not be mixed in streaming mode."
                )
            return (
                concatenate_datasets(all_datasets).shuffle(seed=data_args.seed)
                if shuffle
                else concatenate_datasets(all_datasets)
            )

        elif data_args.mix_strategy.startswith("interleave"):
            if not data_args.streaming:
                logger.warning_rank0_once(
                    "We recommend using `mix_strategy=concat` in non-streaming mode."
                )

            return interleave_datasets(
                datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                seed=data_args.seed,
                stopping_strategy=(
                    "first_exhausted"
                    if data_args.mix_strategy.endswith("under")
                    else "all_exhausted"
                ),
            )
        else:
            raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")

    def split_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        data_args: DataArguments,
        seed: int,
    ) -> DatasetDict:
        """
        Splits the dataset and returns a dataset dict containing train set and validation set.

        Supports both map dataset and iterable dataset.
        """
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
            val_set = dataset.take(int(data_args.val_size))
            train_set = dataset.skip(int(data_args.val_size))
            return DatasetDict({"train": train_set, "validation": val_set})
        else:
            val_size = (
                int(data_args.val_size)
                if data_args.val_size > 1
                else data_args.val_size
            )
            dataset = dataset.train_test_split(test_size=val_size, seed=seed)
            return DatasetDict(
                {"train": dataset["train"], "validation": dataset["test"]}
            )


if __name__ == "__main__":
    data = load_dataset("priyank-m/chinese_text_recognition")

    result = DatasetHandler().merge_dataset([data["test"]], DataArguments)
    breakpoint()
