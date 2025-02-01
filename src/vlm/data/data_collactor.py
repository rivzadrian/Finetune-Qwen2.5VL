import torch

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DataCollatorForQwenVL:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        text, image_inputs, video_inputs = zip(
            *[
                (
                    self.processor.apply_chat_template(
                        batch["text"], tokenize=False, add_generation_prompt=False
                    ),
                    batch["image_inputs"],
                    batch["video_inputs"],
                )
                for batch in features
            ]
        )

        batch = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
