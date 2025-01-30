from datasets import load_dataset
from qwen_vl_utils.vision_process import (
    fetch_image,
    fetch_video,
)
from PIL import Image
import torch
from typing import Optional


class VLMPreprocessor:
    # def convert2dict(self, dataset):
    #     # https://huggingface.co/datasets/priyank-m/chinese_text_recognition
    #     data_dict = []

    #     for idx, data in datatset:

    def process_vision_info(
        self,
        vision_infos: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
    ) -> tuple[
        list[Image.Image] | None,
        list[torch.Tensor | list[Image.Image]] | None,
        Optional[dict],
    ]:
        """
        Example:

        vision_infos = [
            {"image": Image.Oject, "video": Video},
            {"image": Image.Oject, "video": Video},
            {"image": Image.Oject, "video": Video}
        ]

        """
        # Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_input, video_sample_fps = fetch_video(
                    vision_info, return_video_sample_fps=True
                )
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        if return_video_kwargs:
            return image_inputs, video_inputs, {"fps": video_sample_fps_list}
        return image_inputs, video_inputs
