#!/usr/bin/env python3
"""CLI script for sampling from models and calculating rewards."""

import click
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from rl_onoff.backends import get_backend
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.tasks.rewards import create_reward, REWARD_REGISTRY
from rl_onoff.utils.data_loader import load_data


@click.command()
@click.option('--backend-type', type=click.Choice(['huggingface', 'vllm', 'sglang']), 
              default='huggingface', help='Backend type to use')
@click.option('--model-name', required=True, help='Model name or path')
@click.option('--dataset', required=True, help='Path to dataset file')
@click.option('--output', required=True, help='Path to output JSON file')
@click.option('--question-column', default=None, help='Name of question column in dataset')
@click.option('--reference-column', default=None, help='Name of reference column in dataset')
@click.option('--max-length', default=100, type=int, help='Maximum length for generation')
@click.option('--temperature', default=1.0, type=float, help='Sampling temperature')
@click.option('--top-k', default=None, type=int, help='Top-k sampling parameter')
@click.option('--top-p', default=None, type=float, help='Top-p sampling parameter')
@click.option('--num-samples', default=1, type=int, help='Number of samples per prompt')
@click.option('--metrics', default='all', help='Comma-separated list of rewards or "all"')
@click.option('--batch-size', default=None, type=int, help='Batch size for processing')
def main(
    backend_type: str,
    model_name: str,
    dataset: str,
    output: str,
    question_column: str,
    reference_column: str,
    max_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    num_samples: int,
    metrics: str,
    batch_size: int,
):
    """Sample from a model and calculate rewards on the responses."""
    
    # Load dataset
    click.echo(f"Loading dataset from {dataset}...")
    data = load_data(dataset)
    
    # Extract questions and references using specified column names
    question_col = question_column or 'question'
    reference_col = reference_column or 'solution'
    questions = [item.get(question_col, '') for item in data]
    references = [item.get(reference_col, '') for item in data]
    
    # Initialize backend
    click.echo(f"Loading {backend_type} backend with model {model_name}...")
    backend = get_backend(backend_type, model_name=model_name)
    backend.load()
    
    # Initialize sampler
    sampler = Sampler(backend)
    sampling_config = SamplingConfig(
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    
    # Determine which rewards to compute
    if metrics.lower() == 'all':
        reward_names = list(REWARD_REGISTRY.keys())
        # Remove perplexity if backend doesn't support it
        if backend_type != 'huggingface' and 'perplexity' in reward_names:
            reward_names.remove('perplexity')
    else:
        reward_names = [m.strip() for m in metrics.split(',')]
    
    # Create reward instances
    rewards = {}
    for reward_name in reward_names:
        try:
            if reward_name == 'perplexity' and backend_type == 'huggingface':
                rewards[reward_name] = create_reward(reward_name, backend=backend)
            else:
                rewards[reward_name] = create_reward(reward_name)
        except (ImportError, ValueError) as e:
            click.echo(f"Warning: Reward '{reward_name}' unavailable: {e}")
            continue
    
    # Generate samples
    click.echo("Generating samples...")
    predictions = sampler.sample(questions, config=sampling_config)
    
    # Handle multiple samples
    if num_samples > 1:
        # Flatten predictions for reward computation
        flat_predictions = []
        flat_references = []
        for pred_list, ref in zip(predictions, references):
            for pred in pred_list:
                flat_predictions.append(pred)
                flat_references.append(ref)
    else:
        flat_predictions = predictions
        flat_references = references
    
    # Calculate rewards
    click.echo("Calculating rewards...")
    reward_results = {}
    for reward_name, reward in rewards.items():
        try:
            reward_results[reward_name] = reward.compute(flat_predictions, flat_references)
        except Exception as e:
            reward_results[reward_name] = {"error": str(e)}
    
    # Prepare output
    output_data = {
        'model': model_name,
        'backend': backend_type,
        'num_samples': num_samples,
        'reward_results': reward_results,
        'examples': [],
    }
    
    # Add examples
    for i, (question, pred, ref) in enumerate(zip(questions, predictions, references)):
        example = {
            'question': question,
            'prediction': pred,
            'reference': ref,
        }
        output_data['examples'].append(example)
    
    # Save output
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    click.echo(f"Results saved to {output}")
    click.echo(f"\nReward Summary:")
    for reward_name, reward_value in reward_results.items():
        if isinstance(reward_value, dict) and 'error' in reward_value:
            click.echo(f"  {reward_name}: Error - {reward_value['error']}")
        else:
            click.echo(f"  {reward_name}: {reward_value}")


if __name__ == '__main__':
    main()

