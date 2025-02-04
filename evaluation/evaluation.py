from src.inference import VLMConfig, VLMInference
from src.vlm.data.loader import _get_merged_dataset
from typing import Optional, Dict, Any
from src.vlm.hyparams.training_args import TrainingArguments


class OCREvaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        self.config = VLMConfig.from_args(args)
        self.inference = VLMInference(self.config)

    def load_evaluation_dataset(self):
        data = _get_merged_dataset(
            dataset_names=self.config.data_args.dataset,
            model_args=self.config.model_args,
            data_args=self.config.data_args,
            training_args=TrainingArguments(output_dir=None),
        )
        return data

    def evaluate(self):
        pass


# def evaluation(args: Optional[Dict[str, Any]] = None) -> float:
#     # Initialize configuration
#     config = VLMConfig.from_args(args)

#     # Initialize inference pipeline
#     inference = VLMInference(config)

# if __name__ == "__main__":
#     evaluation()
