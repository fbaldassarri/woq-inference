# PyTorch Weights-only-Quantization (WoQ)

Inference scripts for pytorch weights-only-quantization

## TEQ: a trainable equivalent transformation that preserves the FP32 precision in weight-only quantization

### Install

```
conda create -n teq-inference python=3.11

conda activate teq-inference

conda install -c conda-forge gcc

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -re requirements.txt
```

### Usage

```
python teq_inference.py --base <base_model> --model_dir <path-to-woq-TEQ-quantized-model> --weights_file quantized_weight.pt --config_file qconfig.json --prompt "Tell me a joke" --device cpu
```

For example:

```
python teq_inference.py --base meta-llama/Llama-3.2-1B --model_dir ./meta-llama_Llama-3.2-1B-TEQ-int4-gs128-asym --weights_file quantized_weight.pt --config_file qconfig.json --prompt "Tell me a joke" --device cpu
```

