#!/usr/bin/env python3
"""Experimental script 00: Sample responses given task configs and datasets.

This script:
1. Loads a task configuration (task config file)
2. Loads a dataset
3. Formats questions using the task
4. Samples responses from a model
5. Saves results to exp/00-sample/output/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from rl_onoff.backends import create_backend
from rl_onoff.backends.config import BackendConfig
from rl_onoff.sampling import Sampler, SamplingConfig
from rl_onoff.tasks import create_task
from rl_onoff.utils.data_loader import load_data


def main(
    task_config: str,
    dataset: str,
    backend_type: str = "huggingface",
    model_name: str = "gpt2",
    question_column: Optional[str] = None,
    reference_column: Optional[str] = None,
    max_length: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    num_samples: int = 1,
    batch_size: Optional[int] = None,
    output_dir: Optional[str] = None,
):
    """Sample responses given task configs and datasets.
    
    Args:
        task_config: Path to task config file (JSON) or task config dict
        dataset: Path to dataset file
        backend_type: Backend type ('huggingface', 'vllm', 'sglang')
        model_name: Model name or path
        question_column: Name of question column in dataset
        reference_column: Name of reference column in dataset
        max_length: Maximum length for generation
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        num_samples: Number of samples per prompt
        batch_size: Batch size for processing (None for all at once)
        output_dir: Output directory (default: exp/00-sample/output)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "00-sample" / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load task configuration
    print(f"\nLoading task config from {task_config}...")
    task = create_task(task_config)
    print(f"Task: {task.name}")
    print(f"  Template: {task.config.template_type}")
    print(f"  Format: {task.config.format_type}")
    print(f"  Reward: {task.config.reward_type}")
    
    # Load dataset
    print(f"\nLoading dataset from {dataset}...")
    data = load_data(dataset)
    print(f"Loaded {len(data)} examples")
    
    # Extract questions and references
    question_col = question_column or 'question'
    reference_col = reference_column or 'solution'
    questions = [item.get(question_col, '') for item in data]
    
    # Check if reference column exists in data
    has_references = len(data) > 0 and reference_col in data[0]
    references = [item.get(reference_col, '') for item in data] if has_references else None
    
    # Initialize backend
    print(f"\nLoading {backend_type} backend with model {model_name}...")
    backend_config = BackendConfig(
        backend_type=backend_type,
        model_name=model_name
    )
    backend = create_backend(backend_config)
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
    
    # Format queries using task
    print("\nFormatting queries...")
    prompts = []
    for question in tqdm(questions, desc="Formatting"):
        prompt = task.format_query(question)
        prompts.append(prompt)
    
    # Sample responses
    print("\nSampling responses...")
    responses = sampler.sample(prompts, config=sampling_config)
    
    # Handle multiple samples per prompt
    if num_samples > 1:
        # responses is a list of lists
        all_results = []
        for i, (question, ref, response_list) in enumerate(zip(questions, references or [None] * len(questions), responses)):
            for j, response in enumerate(response_list):
                result = {
                    'example_id': i,
                    'sample_id': j,
                    'question': question,
                    'prompt': prompts[i],
                    'response': response,
                }
                if ref is not None:
                    result['reference'] = ref
                all_results.append(result)
    else:
        # responses is a list of strings
        all_results = []
        for i, (question, ref, response) in enumerate(zip(questions, references or [None] * len(questions), responses)):
            result = {
                'example_id': i,
                'question': question,
                'prompt': prompts[i],
                'response': response,
            }
            if ref is not None:
                result['reference'] = ref
            all_results.append(result)
    
    # Extract answers if references are available
    if references:
        print("\nExtracting answers...")
        for result in tqdm(all_results, desc="Extracting"):
            extracted = task.extract_answer(result['response'])
            result['extracted_answer'] = extracted.get('answer')
            result['extracted_reasoning'] = extracted.get('reasoning')
    
    # Save results
    output_file = output_dir / "responses.json"
    print(f"\nSaving results to {output_file}...")
    
    output_data = {
        'task_config': str(task_config),
        'dataset': str(dataset),
        'model': model_name,
        'backend': backend_type,
        'sampling_config': {
            'max_length': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'num_samples': num_samples,
            'batch_size': batch_size,
        },
        'num_examples': len(questions),
        'results': all_results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_results)} results to {output_file}")
    
    # Save summary
    summary_file = output_dir / "summary.json"
    summary = {
        'task_config': str(task_config),
        'dataset': str(dataset),
        'model': model_name,
        'backend': backend_type,
        'num_examples': len(questions),
        'num_samples_per_example': num_samples,
        'total_responses': len(all_results),
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample responses given task configs and datasets")
    parser.add_argument("--task-config", required=True, help="Path to task config file (JSON)")
    parser.add_argument("--dataset", required=True, help="Path to dataset file")
    parser.add_argument("--backend-type", default="huggingface", choices=['huggingface', 'vllm', 'sglang'],
                        help="Backend type")
    parser.add_argument("--model-name", default="gpt2", help="Model name or path")
    parser.add_argument("--question-column", default=None, help="Name of question column in dataset")
    parser.add_argument("--reference-column", default=None, help="Name of reference column in dataset")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for processing")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: exp/00-sample/output)")
    
    args = parser.parse_args()
    
    main(
        task_config=args.task_config,
        dataset=args.dataset,
        backend_type=args.backend_type,
        model_name=args.model_name,
        question_column=args.question_column,
        reference_column=args.reference_column,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

