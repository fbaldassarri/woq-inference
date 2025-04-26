import os
import json
import torch
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def debug_weights_structure(weights_path):
    """Examine the structure of the weights file to help debug loading issues"""
    weights = torch.load(weights_path, map_location="cpu")
    logger.info(f"Type of loaded weights: {type(weights)}")
    if isinstance(weights, dict):
        logger.info(f"Top-level keys: {list(weights.keys())}")
        # Print a few sample keys to understand the structure
        sample_keys = list(weights.keys())[:5]
        for key in sample_keys:
            logger.info(f"Sample key structure: {key} -> {type(weights[key])}")
    return weights

def main():
    parser = argparse.ArgumentParser(description="Run inference with a TEQ-quantized model")
    parser.add_argument("--model_dir", type=str, default=".",
                        help="Directory containing quantized model files")
    parser.add_argument("--weights_file", type=str, default="quantized_weight.pt",
                        help="Name of the quantized weights file")
    parser.add_argument("--config_file", type=str, default="qconfig.json",
                        help="Name of the quantization config file")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Original model name or path (for tokenizer and model architecture)")
    parser.add_argument("--prompt", type=str, default="Once upon a time, a little girl",
                        help="Text prompt for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "xpu"],
                        help="Device to run inference on")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save the generated text to (optional)")
    parser.add_argument("--debug", action="store_true",
                        help="Print additional debug information")
    args = parser.parse_args()

    # Set up paths
    weights_path = os.path.join(args.model_dir, args.weights_file)
    config_path = os.path.join(args.model_dir, args.config_file)
    
    # Check if files exist
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Quantized weights file not found: {weights_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Quantization config file not found: {config_path}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Examine the structure of the weights file
    logger.info(f"Analyzing weights structure from {weights_path}...")
    weights = debug_weights_structure(weights_path)
    
    # Load the base model directly (bypassing TEQ quantization)
    logger.info(f"Loading base model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Print model's state_dict keys for debugging
    if args.debug:
        model_keys = list(model.state_dict().keys())
        logger.info(f"Model has {len(model_keys)} keys in state_dict")
        logger.info(f"Sample model keys: {model_keys[:5]}")
    
    # Check if weights contains 'state_dict' key and adjust accordingly
    if 'state_dict' in weights:
        logger.info("Found 'state_dict' key in weights file, extracting it...")
        weights = weights['state_dict']
    
    # Try to match the weights to the model structure
    try:
        # First attempt: Direct loading
        logger.info("Attempting to load weights directly...")
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} keys in state_dict")
            if args.debug:
                logger.warning(f"Sample missing keys: {missing_keys[:5]}")
        
        if unexpected_keys:
            logger.warning(f"Found {len(unexpected_keys)} unexpected keys in state_dict")
            if args.debug:
                logger.warning(f"Sample unexpected keys: {unexpected_keys[:5]}")
                
        # Validate if we have critical missing keys
        if len(missing_keys) > len(model.state_dict()) * 0.5:
            logger.error("Too many missing keys! Weight loading may have failed")
            
    except Exception as e:
        logger.error(f"Error loading weights: {str(e)}")
        logger.info("Attempting to transform keys to match model structure...")
        
        # Create a transformed state_dict
        transformed_weights = {}
        
        # Try removing 'module.' prefix
        for key in weights:
            if key.startswith('module.'):
                transformed_weights[key[7:]] = weights[key]
            else:
                transformed_weights[key] = weights[key]
        
        # Try loading the transformed weights
        missing_keys, unexpected_keys = model.load_state_dict(transformed_weights, strict=False)
        logger.info(f"After transformation: {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
    
    # Put model in evaluation mode
    model.eval()
    
    # Move model to specified device
    device = args.device
    logger.info(f"Moving model to {device}...")
    model = model.to(device)
    
    # Optimize with IPEX if using Intel hardware
    if device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            logger.info("Optimizing model with IPEX...")
            model = ipex.optimize(model, dtype=torch.float16)
        except ImportError:
            logger.warning("IPEX not available, skipping optimization")
    
    # Run inference
    logger.info(f"Generating text for prompt: '{args.prompt}'")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    logger.info("\nGenerated text:")
    logger.info("-" * 50)
    logger.info(generated_text)
    logger.info("-" * 50)
    
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(generated_text)
        logger.info(f"Generated text saved to {args.output_file}")

if __name__ == "__main__":
    main()
