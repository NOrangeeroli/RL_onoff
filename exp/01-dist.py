#!/usr/bin/env python3
"""Experimental script 01: Extract token-wise distributions from sampled responses.

This script:
1. Loads results from 00-sample.py
2. For each prompt/sample pair, extracts token-wise distributions
3. Splits the sample string according to tokenization
4. Saves results in 01-dist/output/
"""

import json
import sys
import yaml
import numpy as np
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_onoff.backends import create_backend
from rl_onoff.backends.config import BackendConfig
from rl_onoff.distributions import DistributionExtractor


def load_experiment_config(config_path: Optional[str] = None) -> dict:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: 01-dist_config.yaml in same directory)
        
    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent / "01-dist_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
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
        token_strings.append(str(token_str))
    
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
    """Main function to extract token-wise distributions.
    
    Args:
        experiment_config_path: Path to experiment config file (default: 01-dist_config.yaml)
    """
    # Load experiment configuration
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
    backend_config = BackendConfig(
        backend_type=backend_config_dict.get("backend_type", "huggingface"),
        model_name=backend_config_dict.get("model_name", "meta-llama/Llama-3.2-3B"),
        **backend_config_dict.get("backend_kwargs", {})
    )
    backend = create_backend(backend_config)
    backend.load()
    print("Backend loaded successfully!")
    
    # Initialize distribution extractor
    print("\nInitializing distribution extractor...")
    extractor = DistributionExtractor(backend)
    
    # Distribution extraction config
    dist_config = exp_config.get("distribution", {})
    use_logits = dist_config.get("use_logits", False)
    temperature = dist_config.get("temperature", 1.0)
    
    print(f"Distribution config: use_logits={use_logits}, temperature={temperature}")
    
    # Process each example
    print("\n" + "=" * 80)
    print("Extracting distributions...")
    print("=" * 80)
    
    all_dist_results = []
    
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
                
                # Convert distributions to list for JSON serialization
                distributions_list = convert_numpy_to_list(distributions)
                
                sample_dist_result = {
                    "sample_id": sample_id,
                    "response": response,
                    "token_ids": token_ids,
                    "token_strings": token_strings,  # Response split into tokens as tokenized
                    "distributions": distributions_list,  # Shape: (num_tokens, vocab_size)
                    "distribution_shape": list(distributions.shape) if isinstance(distributions, np.ndarray) else None,
                    "num_tokens": len(token_ids)
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
    
    statistics = {
        "num_examples": len(all_dist_results),
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "average_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0,
        "distribution_type": "logits" if use_logits else "probabilities",
        "temperature": temperature,
        "backend": {
            "backend_type": backend_config.backend_type,
            "model_name": backend_config.model_name,
        },
        "input_file": str(input_results_path)
    }
    
    # Save results
    results_file = output_dir / "distributions.json"
    print(f"\nSaving distributions to {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "statistics": statistics,
            "results": all_dist_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_dist_results)} examples to {results_file}")
    
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
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract token-wise distributions from sampled responses")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config file (default: 01-dist_config.yaml)"
    )
    
    args = parser.parse_args()
    main(experiment_config_path=args.config)

