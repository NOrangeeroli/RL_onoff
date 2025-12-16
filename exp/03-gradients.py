#!/usr/bin/env python3
"""Experimental script 03: Extract per-token down-projected LoRA-B gradients from sampled responses.

This script:
1. Loads results from 00-sample.py
2. For each prompt/sample pair, computes per-token LoRA-B gradients
3. Projects gradients to a lower dimension using random projection
4. Saves results in 03-gradients/output/
"""

import json
import os
import sys
import yaml
import numpy as np
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_onoff.backends import create_backend
from rl_onoff.gradients import LoraBGradientProjector


def load_experiment_config(config_path: Optional[str] = None) -> dict:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: experiment_config.yaml in same directory)
        
    Returns:
        Dictionary with configuration for 03-gradients
    """
    if config_path is None:
        config_path = Path(__file__).parent / "experiment_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
    
    # Extract the 03_gradients section
    config = all_configs.get("03_gradients", {})
    if not config:
        raise ValueError("Config file missing '03_gradients' section")
    
    return config


def get_tokenized_strings(backend, text: str) -> List[str]:
    """Get token strings by tokenizing the text.
    
    This splits the string exactly as the tokenizer would tokenize it.
    
    Args:
        backend: Backend instance with tokenizer
        text: Text to tokenize
        
    Returns:
        List of token strings (as they appear in the tokenizer's vocabulary)
    """
    tokenizer = backend.get_tokenizer()
    token_ids = backend.encode(text)
    
    # Convert token IDs to token strings
    token_strings = []
    for token_id in token_ids:
        if hasattr(tokenizer, 'convert_ids_to_tokens'):
            token_str = tokenizer.convert_ids_to_tokens(token_id)
        else:
            # Fallback: decode single token
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        token_str = str(token_str)
        
        # Replace 'Ġ' (special space character) and regular spaces
        token_str = token_str.replace('Ġ', '').replace(' ', '')
        
        token_strings.append(token_str)
    
    return token_strings


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    return obj


def main(
    experiment_config_path: Optional[str] = None
):
    """Main function to extract per-token down-projected LoRA-B gradients.
    
    Args:
        experiment_config_path: Path to experiment config file (default: experiment_config.yaml)
    """
    # Load full config to access global settings
    if experiment_config_path is None:
        config_path = Path(__file__).parent / "experiment_config.yaml"
    else:
        config_path = Path(experiment_config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
    
    # Set up CUDA visible devices from global config (must be done before any CUDA operations)
    global_config = all_configs.get("global", {})
    cuda_config = global_config.get("cuda", {})
    visible_devices = cuda_config.get("visible_devices")
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        print(f"Set CUDA_VISIBLE_DEVICES={visible_devices}")
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        print("CUDA_VISIBLE_DEVICES not set, using all available devices")
    else:
        print(f"Using existing CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Load experiment-specific configuration
    exp_config = load_experiment_config(experiment_config_path)
    
    # Set up output directory
    output_dir_str = exp_config.get("output", {}).get("dir", "exp/03-gradients/output")
    output_dir = (project_root / output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LoRA-B Gradient Extraction and Projection Experiment")
    print("=" * 80)
    
    # Load input results from 00-sample.py
    input_results_path_str = exp_config.get("input", {}).get("results_file", "exp/00-sample/output/results.json")
    input_results_path = project_root / input_results_path_str
    
    print(f"\nLoading results from {input_results_path}...")
    if not input_results_path.exists():
        raise FileNotFoundError(f"Input results file not found: {input_results_path}")
    
    with open(input_results_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    results = input_data.get("results", [])
    print(f"Loaded {len(results)} examples")
    
    # Initialize backend
    backend_config_dict = exp_config.get("backend", {})
    print("\nInitializing backend...")
    backend = create_backend(backend_config_dict)
    backend.load()
    print("Backend loaded successfully!")
    
    # Check if backend has LoRA
    if not hasattr(backend, 'model') or backend.model is None:
        raise RuntimeError("Backend model not loaded")
    
    # Check for LoRA parameters
    has_lora = any('lora' in name.lower() for name, _ in backend.model.named_parameters())
    if not has_lora:
        raise RuntimeError("Backend model does not have LoRA adapters. Please configure LoRA in backend_specific.lora_config")
    
    # Initialize LoRA-B gradient projector
    print("\nInitializing LoRA-B gradient projector...")
    grad_config = exp_config.get("gradient", {})
    proj_dim = grad_config.get("proj_dim", 8192)
    device = grad_config.get("device", "cuda")
    proj_type = grad_config.get("proj_type", "rademacher")
    use_cuda_projector = grad_config.get("use_cuda_projector", False)
    block_size = grad_config.get("block_size", 100)
    seed = grad_config.get("seed", 0)
    cuda_max_batch_size = grad_config.get("cuda_max_batch_size", 32)
    
    print(f"Gradient config: proj_dim={proj_dim}, device={device}, proj_type={proj_type}, "
          f"use_cuda_projector={use_cuda_projector}, seed={seed}")
    
    projector = LoraBGradientProjector(
        backend=backend,
        proj_dim=proj_dim,
        device=device,
        proj_type=proj_type,
        use_cuda_projector=use_cuda_projector,
        block_size=block_size,
        seed=seed,
        cuda_max_batch_size=cuda_max_batch_size,
    )
    print(f"Projector initialized: grad_dim={projector.grad_dim}, proj_dim={projector.proj_dim}")
    
    # Process each example
    print("\n" + "=" * 80)
    print("Extracting gradients...")
    print("=" * 80)
    
    all_grad_results = []
    all_projected_gradients = []  # Store all projected gradients for compact saving
    
    for result in tqdm(results, desc="Processing examples"):
        example_id = result.get("example_id", 0)
        question = result.get("question", "")
        prompt = result.get("prompt", "")
        samples = result.get("samples", [])
        
        example_grad_results = {
            "example_id": example_id,
            "question": question,
            "prompt": prompt,
            "samples": []
        }
        
        # Process each sample for this example
        for sample in samples:
            sample_id = sample.get("sample_id", 0)
            response = sample.get("response", "")
            
            if not response:
                continue
            
            # Compute and project per-token LoRA-B gradients
            try:
                # Compute projected gradients: returns List[np.ndarray] with shape (k, proj_dim)
                projected_grads_list = projector.compute_and_project(
                    prompts=[prompt],  # Use the formatted prompt
                    responses=[response],
                    model_id=0
                )
                
                if len(projected_grads_list) != 1:
                    raise ValueError(f"Expected 1 projected gradient, got {len(projected_grads_list)}")
                
                projected_grads = projected_grads_list[0]  # (k, proj_dim)
                
                # Get token strings (split the response string as it gets tokenized)
                token_strings = get_tokenized_strings(backend, response)
                
                # Get token IDs
                token_ids = backend.encode(response)
                
                # Verify alignment
                if len(token_ids) != len(token_strings):
                    print(f"\nWarning: Token ID count ({len(token_ids)}) != token string count ({len(token_strings)}) "
                          f"for example {example_id}, sample {sample_id}")
                
                if len(token_ids) != projected_grads.shape[0]:
                    print(f"\nWarning: Token count ({len(token_ids)}) != gradient count ({projected_grads.shape[0]}) "
                          f"for example {example_id}, sample {sample_id}")
                
                # Store projected gradients in compact format (will save to NPZ later)
                grad_index = len(all_projected_gradients)
                all_projected_gradients.append(projected_grads)
                
                sample_grad_result = {
                    "sample_id": sample_id,
                    "response": response,
                    "token_ids": token_ids,
                    "token_strings": token_strings,  # Response split into tokens as tokenized
                    "gradient_index": grad_index,  # Index into the gradients array
                    "gradient_shape": list(projected_grads.shape),  # (num_tokens, proj_dim)
                    "num_tokens": len(token_ids),
                    "proj_dim": proj_dim,
                    "grad_dim": projector.grad_dim,  # Original gradient dimension before projection
                }
                
                example_grad_results["samples"].append(sample_grad_result)
                
            except Exception as e:
                print(f"\nWarning: Error processing example {example_id}, sample {sample_id}: {e}")
                traceback.print_exc()
                sample_grad_result = {
                    "sample_id": sample_id,
                    "response": response,
                    "error": str(e)
                }
                example_grad_results["samples"].append(sample_grad_result)
        
        all_grad_results.append(example_grad_results)
    
    # Calculate statistics
    total_samples = sum(len(r["samples"]) for r in all_grad_results)
    total_tokens = sum(
        sum(s.get("num_tokens", 0) for s in r["samples"])
        for r in all_grad_results
    )
    
    # Get backend config info from the dict
    backend_type_str = backend_config_dict.get("backend_type", "huggingface")
    backend_model_name = backend_config_dict.get("model_name", "unknown")
    
    statistics = {
        "num_examples": len(all_grad_results),
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "average_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
        "proj_dim": proj_dim,
        "grad_dim": projector.grad_dim,
        "proj_type": proj_type,
        "use_cuda_projector": use_cuda_projector,
        "seed": seed,
        "backend": {
            "backend_type": backend_type_str,
            "model_name": backend_model_name,
        },
        "input_file": str(input_results_path)
    }
    
    # Save projected gradients in compact NPZ format
    gradients_file = output_dir / "gradients.npz"
    print(f"\nSaving projected gradients to {gradients_file}...")
    if all_projected_gradients:
        # Save all gradients as a dictionary of arrays
        # Key format: "grad_{index}" for each gradient
        gradients_dict = {f"grad_{i}": grad for i, grad in enumerate(all_projected_gradients)}
        np.savez_compressed(gradients_file, **gradients_dict)
        print(f"Saved {len(all_projected_gradients)} projected gradients to {gradients_file}")
    else:
        print("No gradients to save")
    
    # Save metadata (without gradients) in JSON
    results_file = output_dir / "gradients.json"
    print(f"\nSaving metadata to {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "statistics": statistics,
            "results": all_grad_results,
            "gradients_file": str(gradients_file.name),  # Store relative filename
            "num_gradients": len(all_projected_gradients)
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_grad_results)} examples metadata to {results_file}")
    
    # Save statistics
    stats_file = output_dir / "statistics.json"
    print(f"\nSaving statistics to {stats_file}...")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to {stats_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total examples: {len(all_grad_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per sample: {total_tokens / total_samples if total_samples > 0 else 0:.1f}")
    print(f"Projection dimension: {proj_dim}")
    print(f"Original gradient dimension: {projector.grad_dim}")
    print(f"Compression ratio: {projector.grad_dim / proj_dim:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract per-token down-projected LoRA-B gradients from sampled responses")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config file (default: experiment_config.yaml)"
    )
    
    args = parser.parse_args()
    main(experiment_config_path=args.config)

