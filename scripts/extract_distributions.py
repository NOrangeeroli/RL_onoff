#!/usr/bin/env python3
"""CLI script for extracting token-wise distributions from models."""

import click
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from rl_onoff.backends import get_backend
from rl_onoff.distributions import DistributionExtractor
from rl_onoff.utils.data_loader import load_data


@click.command()
@click.option('--backend-type', type=click.Choice(['huggingface', 'vllm', 'sglang']), 
              default='huggingface', help='Backend type to use')
@click.option('--model-name', required=True, help='Model name or path')
@click.option('--dataset', required=True, help='Path to dataset file')
@click.option('--output', required=True, help='Path to output file (JSON or NPZ)')
@click.option('--question-column', default=None, help='Name of question column in dataset')
@click.option('--solution-column', default=None, help='Name of solution column in dataset')
@click.option('--use-logits', is_flag=True, help='Extract logits instead of probabilities')
@click.option('--temperature', default=1.0, type=float, help='Temperature for probability normalization')
@click.option('--format', type=click.Choice(['json', 'npz']), default='npz',
              help='Output format (json or npz)')
def main(
    backend_type: str,
    model_name: str,
    dataset: str,
    output: str,
    question_column: str,
    solution_column: str,
    use_logits: bool,
    temperature: float,
    format: str,
):
    """Extract token-wise distributions from a model for question-solution pairs."""
    
    # Load dataset
    click.echo(f"Loading dataset from {dataset}...")
    data = load_data(
        dataset,
        question_column=question_column or 'question',
        solution_column=solution_column or 'solution',
    )
    
    questions = [item['question'] for item in data]
    solutions = [item.get('solution', '') for item in data]
    
    # Initialize backend
    click.echo(f"Loading {backend_type} backend with model {model_name}...")
    backend = get_backend(backend_type, model_name=model_name)
    backend.load()
    
    # Initialize extractor
    extractor = DistributionExtractor(backend)
    
    # Extract distributions
    click.echo("Extracting distributions...")
    all_distributions = []
    all_token_ids = []
    
    for question, solution in tqdm(zip(questions, solutions), total=len(questions)):
        try:
            dist, token_ids = extractor.extract_distributions(
                question=question,
                solution=solution,
                use_logits=use_logits,
                temperature=temperature,
                return_token_ids=True,
            )
            all_distributions.append(dist)
            all_token_ids.append(token_ids)
        except Exception as e:
            click.echo(f"Error processing question-solution pair: {e}")
            all_distributions.append(None)
            all_token_ids.append(None)
    
    # Save output
    click.echo(f"Saving results to {output}...")
    output_path = Path(output)
    
    if format == 'npz':
        # Save as compressed numpy arrays
        valid_data = {
            'distributions': [d for d in all_distributions if d is not None],
            'token_ids': [t for t in all_token_ids if t is not None],
        }
        np.savez_compressed(output_path, **valid_data)
    else:
        # Save as JSON (convert arrays to lists, may be large)
        output_data = {
            'model': model_name,
            'backend': backend_type,
            'use_logits': use_logits,
            'temperature': temperature,
            'distributions': [],
            'token_ids': [],
        }
        
        for dist, token_ids in zip(all_distributions, all_token_ids):
            if dist is not None:
                output_data['distributions'].append(dist.tolist())
                output_data['token_ids'].append(token_ids)
            else:
                output_data['distributions'].append(None)
                output_data['token_ids'].append(None)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    
    click.echo(f"Results saved to {output}")
    click.echo(f"Extracted distributions for {len([d for d in all_distributions if d is not None])} question-solution pairs")


if __name__ == '__main__':
    main()

