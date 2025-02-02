# Finetune-Qwen2.5-VL


TBD


### Quick Start 
- SFT
```python
python src/train.py config/vlm_config.yaml 
```

- Merge LoRA
```python
python src/merge_model.py config/vlm_merge_adapter_config.yaml 
```

- Inference
```python
python src/inference.py config/vlm_inference_config.yaml
```


### Acknowledge
This project is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main). Special thanks to the original authors and contributors!