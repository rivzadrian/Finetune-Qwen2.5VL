# Finetune-Qwen2.5-VL


TBD


### Quick Start 
- SFT
```python
python src/train.py vlm_config.yaml 
```

- Merge LoRA
```python
python src/merge_model.py vlm_merge_adapter_config.yaml 
```

- Inference
```python
python src/inference.py vlm_inference_config.yaml
```


### Acknowledge
This project is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main). Special thanks to the original authors and contributors!