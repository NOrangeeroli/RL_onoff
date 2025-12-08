#!/usr/bin/env python3
"""CLI script for calculating token-level divergence between models."""

import click
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from rl_onoff.backends import get_backend
from rl_onoff.divergence import DivergenceCalculator
from rl_onoff.utils.data_loader import load_data


@click.command()
@click.option('--backend-type-a', type=click.Choice(['huggingface', 'vllm', 'sglang']), 
              default='huggingface', help='Backend type for model A')
@click.option('--model-name-a', required=True, help='Model A name or path')
@click.option('--backend-type-b', type=click.Choice(['huggingface', 'vllm', 'sglang']), 
              default='huggingface', help='Backend type for model B')
@click.option('--model-name-b', required=True, help='Model B name or path')
@click.option('--dataset', required=True, help='Path to dataset file')
@click.option('--output', required=True, help='Path to output JSON file')
@click.option('--question-column', default=None, help='Name of question column in dataset')
@click.option('--solution-column', default=None, help='Name of solution column in dataset')
@click.option('--divergence-type', type=click.Choice(['kl', 'js', 'both']), 
              default='both', help='Type of divergence to compute')
@click.option('--use-logits', is_flag=True, help='Use logits instead of probabilities')
@click.option('--temperature', default=1.0, type=float, help='Temperature for probability normalization')
def main(
    backend_type_a: str,
    model_name_a: str,
    backend_type_b: str,
    model_name_b: str,
    dataset: str,
    output: str,
    question_column: str,
    solution_column: str,
    divergence_type: str,
    use_logits: bool,
    temperature: float,
):
    """Calculate token-level divergence between two models for question-solution pairs."""
    
    # Load dataset
    click.echo(f"Loading dataset from {dataset}...")
    data = load_data(dataset)
    
    # Extract questions and solutions using specified column names
    question_col = question_column or 'question'
    solution_col = solution_column or 'solution'
    questions = [item.get(question_col, '') for item in data]
    solutions = [item.get(solution_col, '') for item in data]
    
    # Initialize backends
    click.echo(f"Loading model A: {backend_type_a} backend with {model_name_a}...")
    backend_a = get_backend(backend_type_a, model_name=model_name_a)
    backend_a.load()
    
    click.echo(f"Loading model B: {backend_type_b} backend with {model_name_b}...")
    backend_b = get_backend(backend_type_b, model_name=model_name_b)
    backend_b.load()
    
    # Initialize divergence calculator
    calculator = DivergenceCalculator()
    
    # Calculate divergences
    click.echo("Calculating divergences...")
    all_results = []
    
    for question, solution in tqdm(zip(questions, solutions), total=len(questions)):
        try:
            result = calculator.compute_divergence_for_solutions(
                question=question,
                solution=solution,
                backend_a=backend_a,
                backend_b=backend_b,
                divergence_type=divergence_type,
                use_logits=use_logits,
                temperature=temperature,
            )
            
            # Convert numpy arrays to lists for JSON serialization
            result_dict = {
                'question': question,
                'solution': solution,
            }
            
            if 'kl' in result:
                result_dict['kl_divergence'] = result['kl'].tolist()
                result_dict['kl_mean'] = float(np.mean(result['kl']))
                result_dict['kl_max'] = float(np.max(result['kl']))
            
            if 'js' in result:
                result_dict['js_divergence'] = result['js'].tolist()
                result_dict['js_mean'] = float(np.mean(result['js']))
                result_dict['js_max'] = float(np.max(result['js']))
            
            all_results.append(result_dict)
            
        except Exception as e:
            click.echo(f"Error processing question-solution pair: {e}")
            all_results.append({
                'question': question,
                'solution': solution,
                'error': str(e),
            })
    
    # Compute summary statistics
    summary = {
        'model_a': model_name_a,
        'backend_a': backend_type_a,
        'model_b': model_name_b,
        'backend_b': backend_type_b,
        'divergence_type': divergence_type,
        'num_pairs': len(all_results),
    }
    
    if divergence_type in ['kl', 'both']:
        kl_means = [r.get('kl_mean', 0) for r in all_results if 'kl_mean' in r]
        if kl_means:
            summary['kl_statistics'] = {
                'mean': float(np.mean(kl_means)),
                'std': float(np.std(kl_means)),
                'min': float(np.min(kl_means)),
                'max': float(np.max(kl_means)),
            }
    
    if divergence_type in ['js', 'both']:
        js_means = [r.get('js_mean', 0) for r in all_results if 'js_mean' in r]
        if js_means:
            summary['js_statistics'] = {
                'mean': float(np.mean(js_means)),
                'std': float(np.std(js_means)),
                'min': float(np.min(js_means)),
                'max': float(np.max(js_means)),
            }
    
    # Prepare output
    output_data = {
        'summary': summary,
        'results': all_results,
    }
    
    # Save output
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    click.echo(f"Results saved to {output}")
    click.echo(f"\nSummary:")
    click.echo(f"  Processed {summary['num_pairs']} question-solution pairs")
    if 'kl_statistics' in summary:
        stats = summary['kl_statistics']
        click.echo(f"\n  KL Divergence Statistics:")
        click.echo(f"    Mean: {stats['mean']:.4f}")
        click.echo(f"    Std:  {stats['std']:.4f}")
        click.echo(f"    Min:  {stats['min']:.4f}")
        click.echo(f"    Max:  {stats['max']:.4f}")
    if 'js_statistics' in summary:
        stats = summary['js_statistics']
        click.echo(f"\n  JS Divergence Statistics:")
        click.echo(f"    Mean: {stats['mean']:.4f}")
        click.echo(f"    Std:  {stats['std']:.4f}")
        click.echo(f"    Min:  {stats['min']:.4f}")
        click.echo(f"    Max:  {stats['max']:.4f}")


if __name__ == '__main__':
    main()

