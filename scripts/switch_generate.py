#!/usr/bin/env python3
"""CLI script for generating with conditional model switching."""

import click
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from rl_onoff.backends import get_backend
from rl_onoff.switching import ModelSwitcher
from rl_onoff.utils.data_loader import load_data


@click.command()
@click.option('--backend-type-a', type=click.Choice(['huggingface', 'vllm', 'sglang']), 
              default='huggingface', help='Backend type for model A')
@click.option('--model-name-a', required=True, help='Model A name or path')
@click.option('--backend-type-b', type=click.Choice(['huggingface', 'vllm', 'sglang']), 
              default='huggingface', help='Backend type for model B')
@click.option('--model-name-b', required=True, help='Model B name or path')
@click.option('--dataset', default=None, help='Path to dataset file (or use --question)')
@click.option('--question', default=None, help='Single question to process')
@click.option('--output', required=True, help='Path to output JSON file')
@click.option('--question-column', default=None, help='Name of question column in dataset')
@click.option('--divergence-type', type=click.Choice(['kl', 'js']), 
              default='js', help='Type of divergence to use for switching')
@click.option('--threshold', default=0.5, type=float, 
              help='Divergence threshold for switching from A to B')
@click.option('--switch-back-threshold', default=None, type=float,
              help='Divergence threshold for switching back to A (default: threshold / 2)')
@click.option('--max-new-tokens', default=100, type=int, help='Maximum tokens to generate')
@click.option('--temperature', default=1.0, type=float, help='Sampling temperature')
@click.option('--top-k', default=None, type=int, help='Top-k sampling parameter')
@click.option('--top-p', default=None, type=float, help='Top-p sampling parameter')
def main(
    backend_type_a: str,
    model_name_a: str,
    backend_type_b: str,
    model_name_b: str,
    dataset: str,
    question: str,
    output: str,
    question_column: str,
    divergence_type: str,
    threshold: float,
    switch_back_threshold: float,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    """Generate responses with conditional model switching based on divergence thresholds."""
    
    # Get questions
    if question:
        questions = [question]
    elif dataset:
        click.echo(f"Loading dataset from {dataset}...")
        data = load_data(
            dataset,
            question_column=question_column or 'question',
            solution_column=None,
        )
        questions = [item['question'] for item in data]
    else:
        raise click.BadParameter("Either --dataset or --question must be provided")
    
    # Initialize backends
    click.echo(f"Loading model A: {backend_type_a} backend with {model_name_a}...")
    backend_a = get_backend(backend_type_a, model_name=model_name_a)
    backend_a.load()
    
    click.echo(f"Loading model B: {backend_type_b} backend with {model_name_b}...")
    backend_b = get_backend(backend_type_b, model_name=model_name_b)
    backend_b.load()
    
    # Initialize switcher
    switcher = ModelSwitcher(
        backend_a=backend_a,
        backend_b=backend_b,
        divergence_type=divergence_type,
        threshold=threshold,
        switch_back_threshold=switch_back_threshold,
    )
    
    # Generate with switching
    click.echo("Generating responses with conditional switching...")
    all_results = []
    
    for question_text in tqdm(questions, total=len(questions)):
        try:
            response, switch_points = switcher.generate_with_switching(
                question=question_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_switch_points=True,
            )
            
            result = {
                'question': question_text,
                'response': response,
                'num_switches': len(switch_points),
                'switch_points': switch_points,
            }
            all_results.append(result)
            
        except Exception as e:
            click.echo(f"Error processing question: {e}")
            all_results.append({
                'question': question_text,
                'error': str(e),
            })
    
    # Prepare output
    output_data = {
        'model_a': model_name_a,
        'backend_a': backend_type_a,
        'model_b': model_name_b,
        'backend_b': backend_type_b,
        'divergence_type': divergence_type,
        'threshold': threshold,
        'switch_back_threshold': switcher.switch_back_threshold,
        'max_new_tokens': max_new_tokens,
        'num_questions': len(questions),
        'results': all_results,
    }
    
    # Compute summary statistics
    total_switches = sum(r.get('num_switches', 0) for r in all_results)
    output_data['summary'] = {
        'total_switches': total_switches,
        'avg_switches_per_question': total_switches / len(all_results) if all_results else 0,
    }
    
    # Save output
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    click.echo(f"Results saved to {output}")
    click.echo(f"\nSummary:")
    click.echo(f"  Processed {len(questions)} questions")
    click.echo(f"  Total switches: {total_switches}")
    click.echo(f"  Average switches per question: {output_data['summary']['avg_switches_per_question']:.2f}")


if __name__ == '__main__':
    main()

