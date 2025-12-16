#!/usr/bin/env python3
"""Experimental script 01: Extract token-wise distributions from sampled responses.

This script:
1. Loads results from 00-sample.py
2. For each prompt/sample pair, extracts token-wise distributions
3. Splits the sample string according to tokenization
4. Saves results in 01-dist/output/
"""

import json
import os
import sys
import yaml
import numpy as np
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Union, Union
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_onoff.backends import create_backend
from rl_onoff.distributions import DistributionExtractor


def load_experiment_config(config_path: Optional[str] = None) -> dict:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: experiment_config.yaml in same directory)
        
    Returns:
        Dictionary with configuration for 01-dist
    """
    if config_path is None:
        config_path = Path(__file__).parent / "experiment_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
    
    # Extract the 01_dist section
    config = all_configs.get("01_dist", {})
    if not config:
        raise ValueError("Config file missing '01_dist' section")
    
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


def calculate_entropy(probs: np.ndarray) -> float:
    """Calculate entropy from a probability distribution.
    
    Args:
        probs: Probability distribution array (should sum to 1)
        
    Returns:
        Entropy value in nats (natural log)
    """
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    # Calculate entropy: H = -Σ p * log(p)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def calculate_per_token_entropy(distributions: np.ndarray, use_logits: bool = False, temperature: float = 1.0) -> List[float]:
    """Calculate entropy for each token position from distributions.
    
    Args:
        distributions: Distribution array with shape (num_tokens, vocab_size)
                       Can be logits or probabilities
        use_logits: If True, distributions are logits; if False, they are probabilities
        temperature: Temperature for converting logits to probabilities (if use_logits=True)
        
    Returns:
        List of entropy values, one per token position
    """
    if use_logits:
        # Convert logits to probabilities using temperature
        scaled_logits = distributions / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    else:
        probs = distributions
    
    # Calculate entropy for each token position
    per_token_entropy = []
    for i in range(probs.shape[0]):
        entropy = calculate_entropy(probs[i])
        per_token_entropy.append(entropy)
    
    return per_token_entropy


def main(
    experiment_config_path: Optional[str] = None
):
    """Main function to extract token-wise distributions.
    
    Args:
        experiment_config_path: Path to experiment config file (default: 01-dist_config.yaml)
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
    output_dir_str = exp_config.get("output", {}).get("dir", "exp/01-dist/output")
    output_dir = (project_root / output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Token Distribution Extraction Experiment")
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
    # Just pass the dict directly - create_backend handles everything!
    backend = create_backend(backend_config_dict)
    backend.load()
    print("Backend loaded successfully!")
    
    # Initialize distribution extractor
    print("\nInitializing distribution extractor...")
    extractor = DistributionExtractor(backend)
    
    # Distribution extraction config
    dist_config = exp_config.get("distribution", {})
    use_logits = dist_config.get("use_logits", False)
    temperature = dist_config.get("temperature", 1.0)
    top_k = dist_config.get("top_k", 1000)  # Number of top entries to store per token
    
    print(f"Distribution config: use_logits={use_logits}, temperature={temperature}, top_k={top_k}")
    
    # Process each example
    print("\n" + "=" * 80)
    print("Extracting distributions...")
    print("=" * 80)
    
    all_dist_results = []
    all_distributions = []  # Store all distributions for compact saving
    
    for result in tqdm(results, desc="Processing examples"):
        example_id = result.get("example_id", 0)
        question = result.get("question", "")
        prompt = result.get("prompt", "")
        samples = result.get("samples", [])
        
        example_dist_results = {
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
            
            # Extract token-wise distributions
            try:
                distributions, token_ids = extractor.extract_distributions(
                    question=prompt,  # Use the formatted prompt
                    solution=response,
                    use_logits=use_logits,
                    temperature=temperature,
                    return_token_ids=True
                )
                
                # Get token strings (split the response string as it gets tokenized)
                # This should match the token_ids from extract_distributions
                token_strings = get_tokenized_strings(backend, response)
                
                # Verify token_ids and token_strings are aligned
                # (They should be, but let's make sure)
                if len(token_ids) != len(token_strings):
                    print(f"\nWarning: Token ID count ({len(token_ids)}) != token string count ({len(token_strings)}) "
                          f"for example {example_id}, sample {sample_id}")
                
                # Calculate per-token entropy
                per_token_entropy = calculate_per_token_entropy(
                    distributions,
                    use_logits=use_logits,
                    temperature=temperature
                )
                
                # Extract top-k entries for each token position
                # distributions shape: (num_tokens, vocab_size)
                num_tokens = distributions.shape[0]
                vocab_size = distributions.shape[1]
                top_k_actual = min(top_k, vocab_size)
                
                # For each token position, get top-k indices and values
                top_k_indices = np.zeros((num_tokens, top_k_actual), dtype=np.int32)
                top_k_values = np.zeros((num_tokens, top_k_actual), dtype=distributions.dtype)
                
                for token_idx in range(num_tokens):
                    # Get top-k indices (largest values)
                    top_indices = np.argsort(distributions[token_idx])[-top_k_actual:][::-1]
                    top_k_indices[token_idx] = top_indices
                    top_k_values[token_idx] = distributions[token_idx][top_indices]
                
                # Store top-k distribution in compact format (will save to NPZ later)
                dist_index = len(all_distributions)
                all_distributions.append({
                    "indices": top_k_indices,
                    "values": top_k_values,
                    "vocab_size": vocab_size,
                    "top_k": top_k_actual
                })
                
                sample_dist_result = {
                    "sample_id": sample_id,
                    "response": response,
                    "token_ids": token_ids,
                    "token_strings": token_strings,  # Response split into tokens as tokenized
                    "distribution_index": dist_index,  # Index into the distributions array
                    "distribution_shape": [num_tokens, vocab_size],  # Original shape (num_tokens, vocab_size)
                    "top_k": top_k_actual,  # Number of top entries stored per token
                    "num_tokens": len(token_ids),
                    "per_token_entropy": per_token_entropy  # Entropy for each token position
                }
                
                example_dist_results["samples"].append(sample_dist_result)
                
            except Exception as e:
                print(f"\nWarning: Error processing example {example_id}, sample {sample_id}: {e}")
                traceback.print_exc()
                sample_dist_result = {
                    "sample_id": sample_id,
                    "response": response,
                    "error": str(e)
                }
                example_dist_results["samples"].append(sample_dist_result)
        
        all_dist_results.append(example_dist_results)
    
    # Calculate statistics
    total_samples = sum(len(r["samples"]) for r in all_dist_results)
    total_tokens = sum(
        sum(s.get("num_tokens", 0) for s in r["samples"])
        for r in all_dist_results
    )
    
    # Get backend config info from the dict
    backend_type_str = backend_config_dict.get("backend_type", "huggingface")
    backend_model_name = backend_config_dict.get("model_name", "unknown")
    
    statistics = {
        "num_examples": len(all_dist_results),
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "average_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
        "distribution_type": "logits" if use_logits else "probabilities",
        "temperature": temperature,
        "top_k": top_k,
        "backend": {
            "backend_type": backend_type_str,
            "model_name": backend_model_name,
        },
        "input_file": str(input_results_path)
    }
    
    # Save distributions in compact NPZ format (top-k only)
    distributions_file = output_dir / "distributions.npz"
    print(f"\nSaving top-{top_k} distributions to {distributions_file}...")
    if all_distributions:
        # Save all distributions as a dictionary of arrays
        # Each distribution has "indices" and "values" arrays
        distributions_dict = {}
        for i, dist_dict in enumerate(all_distributions):
            distributions_dict[f"dist_{i}_indices"] = dist_dict["indices"]
            distributions_dict[f"dist_{i}_values"] = dist_dict["values"]
            distributions_dict[f"dist_{i}_vocab_size"] = np.array([dist_dict["vocab_size"]], dtype=np.int32)
            distributions_dict[f"dist_{i}_top_k"] = np.array([dist_dict["top_k"]], dtype=np.int32)
        np.savez_compressed(distributions_file, **distributions_dict)
        print(f"Saved {len(all_distributions)} top-{top_k} distributions to {distributions_file}")
    else:
        print("No distributions to save")
    
    # Save metadata (without distributions) in JSON
    results_file = output_dir / "distributions.json"
    print(f"\nSaving metadata to {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "statistics": statistics,
            "results": all_dist_results,
            "distributions_file": str(distributions_file.name),  # Store relative filename
            "num_distributions": len(all_distributions)
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_dist_results)} examples metadata to {results_file}")
    
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
    print(f"Total examples: {len(all_dist_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per sample: {total_tokens / total_samples if total_samples > 0 else 0:.1f}")
    print(f"Distribution type: {'logits' if use_logits else 'probabilities'}")
    print(f"Top-k stored per token: {top_k}")
    print("=" * 80)


def load_distributions(
    json_file: Union[str, Path],
    distributions_file: Optional[Union[str, Path]] = None
) -> Dict:
    """Load distributions from JSON metadata and NPZ distributions file.
    
    This function reconstructs the full data structure by loading distributions
    from the compact NPZ file and merging them with the JSON metadata.
    
    Args:
        json_file: Path to the distributions.json metadata file
        distributions_file: Path to the distributions.npz file (if None, inferred from json_file)
        
    Returns:
        Dictionary with the same structure as the original distributions.json,
        but with 'distributions' arrays included in each sample (as lists)
    """
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load JSON metadata
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine distributions file path
    if distributions_file is None:
        # Try to get from metadata, or infer from JSON file location
        dist_filename = data.get("distributions_file", "distributions.npz")
        distributions_file = json_path.parent / dist_filename
    else:
        distributions_file = Path(distributions_file)
    
    if not distributions_file.exists():
        raise FileNotFoundError(f"Distributions file not found: {distributions_file}")
    
    # Load distributions from NPZ
    print(f"Loading top-k distributions from {distributions_file}...")
    distributions_data = np.load(distributions_file)
    
    # Count number of distributions (each has indices and values)
    num_dists = len([k for k in distributions_data.files if k.endswith("_indices")])
    print(f"Loaded {num_dists} distributions")
    
    # Reconstruct top-k structure by adding distributions to each sample
    results = data.get("results", [])
    for example_result in results:
        for sample in example_result.get("samples", []):
            if "error" in sample:
                # Skip samples with errors
                continue
            
            dist_index = sample.get("distribution_index")
            if dist_index is not None:
                # Load the top-k distribution (indices and values)
                indices_key = f"dist_{dist_index}_indices"
                values_key = f"dist_{dist_index}_values"
                vocab_size_key = f"dist_{dist_index}_vocab_size"
                top_k_key = f"dist_{dist_index}_top_k"
                
                if indices_key in distributions_data and values_key in distributions_data:
                    sample["top_k_distribution"] = {
                        "indices": distributions_data[indices_key].tolist(),
                        "values": distributions_data[values_key].tolist(),
                        "vocab_size": int(distributions_data[vocab_size_key][0]) if vocab_size_key in distributions_data else None,
                        "top_k": int(distributions_data[top_k_key][0]) if top_k_key in distributions_data else None
                    }
                else:
                    print(f"Warning: Distribution index {dist_index} not found in NPZ file")
    
    return data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract token-wise distributions from sampled responses")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config file (default: experiment_config.yaml)"
    )
    
    args = parser.parse_args()
    main(experiment_config_path=args.config)

