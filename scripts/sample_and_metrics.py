#!/usr/bin/env python3
"""CLI script for sampling from models and calculating metrics."""

import click
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from rl_onoff.backends import get_backend
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.metrics import MetricRegistry
from rl_onoff.metrics.builtin import (
    PerplexityMetric,
    BLEUMetric,
    ROUGEMetric,
    ExactMatchMetric,
)
from rl_onoff.utils.data_loader import load_data


@click.command()
@click.option('--backend-type', type=click.Choice(['huggingface', 'vllm', 'sglang']), 
              default='huggingface', help='Backend type to use')
@click.option('--model-name', required=True, help='Model name or path')
@click.option('--dataset', required=True, help='Path to dataset file')
@click.option('--output', required=True, help='Path to output JSON file')
@click.option('--question-column', default=None, help='Name of question column in dataset')
@click.option('--reference-column', default=None, help='Name of reference column in dataset')
@click.option('--max-new-tokens', default=100, type=int, help='Maximum tokens to generate')
@click.option('--temperature', default=1.0, type=float, help='Sampling temperature')
@click.option('--top-k', default=None, type=int, help='Top-k sampling parameter')
@click.option('--top-p', default=None, type=float, help='Top-p sampling parameter')
@click.option('--num-samples', default=1, type=int, help='Number of samples per prompt')
@click.option('--metrics', default='all', help='Comma-separated list of metrics or "all"')
@click.option('--batch-size', default=None, type=int, help='Batch size for processing')
def main(
    backend_type: str,
    model_name: str,
    dataset: str,
    output: str,
    question_column: str,
    reference_column: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    num_samples: int,
    metrics: str,
    batch_size: int,
):
    """Sample from a model and calculate metrics on the responses."""
    
    # Load dataset
    click.echo(f"Loading dataset from {dataset}...")
    data = load_data(
        dataset,
        question_column=question_column or 'question',
        solution_column=reference_column or 'solution',
    )
    
    questions = [item['question'] for item in data]
    references = [item.get('solution', '') for item in data]
    
    # Initialize backend
    click.echo(f"Loading {backend_type} backend with model {model_name}...")
    backend = get_backend(backend_type, model_name=model_name)
    backend.load()
    
    # Initialize sampler
    sampler = Sampler(backend)
    sampling_config = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_samples=num_samples,
    )
    
    # Initialize metrics
    metric_registry = MetricRegistry()
    
    # Register built-in metrics
    if backend_type == 'huggingface':
        metric_registry.register(PerplexityMetric(backend=backend))
    
    try:
        metric_registry.register(BLEUMetric())
        metric_registry.register(ROUGEMetric())
    except ImportError as e:
        click.echo(f"Warning: Some metrics unavailable: {e}")
    
    metric_registry.register(ExactMatchMetric())
    
    # Determine which metrics to compute
    if metrics.lower() == 'all':
        metric_names = None
    else:
        metric_names = [m.strip() for m in metrics.split(',')]
    
    # Generate samples
    click.echo("Generating samples...")
    if batch_size:
        predictions = sampler.sample_batch(
            questions,
            config=sampling_config,
            batch_size=batch_size,
        )
    else:
        predictions = sampler.sample(questions, config=sampling_config)
    
    # Handle multiple samples
    if num_samples > 1:
        # Flatten predictions for metric computation
        flat_predictions = []
        flat_references = []
        for pred_list, ref in zip(predictions, references):
            for pred in pred_list:
                flat_predictions.append(pred)
                flat_references.append(ref)
    else:
        flat_predictions = predictions
        flat_references = references
    
    # Calculate metrics
    click.echo("Calculating metrics...")
    metric_results = metric_registry.compute_all(
        flat_predictions,
        flat_references,
        metric_names=metric_names,
    )
    
    # Prepare output
    output_data = {
        'model': model_name,
        'backend': backend_type,
        'num_samples': num_samples,
        'metric_results': metric_results,
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
    click.echo(f"\nMetric Summary:")
    for metric_name, metric_value in metric_results.items():
        if isinstance(metric_value, dict) and 'error' in metric_value:
            click.echo(f"  {metric_name}: Error - {metric_value['error']}")
        else:
            click.echo(f"  {metric_name}: {metric_value}")


if __name__ == '__main__':
    main()

