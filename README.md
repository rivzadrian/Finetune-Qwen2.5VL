# Finetune-Qwen2.5-VL

A comprehensive toolkit for finetuning the Qwen2.5-VL (Visual Language) model using LoRA. This project provides easy-to-use scripts for Supervised Fine-Tuning (SFT), LoRA merging, and inference.

## Requirements

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.37.0
peft>=0.7.0
accelerate>=0.21.0
```

## Installation

```bash
git clone https://github.com/sandy1990418/Finetune-Qwen2.5-VL.git
cd Finetune-Qwen2.5-VL
pip install -r requirements.txt
```

## Usage

### 1.1 Supervised Fine-Tuning (SFT) - Single GPU

Run the following command to start the fine-tuning process:

```bash
python src/train.py config/vlm_config.yaml

or

python main.py config/vlm_config.yaml
```

The `vlm_config.yaml` should contain your training configurations such as:
- Model parameters
- Training hyperparameters
- Dataset configurations
- LoRA settings

### 1.2 Supervised Fine-Tuning (SFT) - Multiple GPU

Run the following command to start the fine-tuning process:

```bash
python main.py config/vlm_config.yaml config/accelerate.yaml
```

The `vlm_config.yaml` should contain your training configurations and  `accelerate.yaml` should contain accelerate configurations. If you want to use multiple GPUs, setting `use_accelerate: true` in `vlm_config.yaml`.

### 2. Merge LoRA Weights

After training, merge the LoRA weights with the base model:

```bash
python src/merge_model.py config/vlm_merge_adapter_config.yaml
```

### 3. Inference

Run inference with your fine-tuned model:

```bash
python src/inference.py config/vlm_inference_config.yaml
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledge
This project is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main). Special thanks to the original authors and contributors!
