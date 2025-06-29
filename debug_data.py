from src.vlm.data.loader import get_dataset
from src.vlm.hyparams.model_args import ModelArguments
from src.vlm.hyparams.finetune_args import FinetuningArguments

# Example usage
if __name__ == "__main__":
    # Load your dataset with existing utils
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct")
    finetuning_args = FinetuningArguments(stage="sft")

    dataset = get_dataset(model_args, finetuning_args, split="train")

    # Print first few samples for debugging
    for i, sample in enumerate(dataset):
        print(f"Sample {i}:")
        print(sample)
        print("-" * 50)
        if i >= 5:
            break