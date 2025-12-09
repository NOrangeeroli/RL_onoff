#!/usr/bin/env python3
"""Experimental script 00: Sample responses for GSM8K dataset using MathTask.

This script:
1. Loads GSM8K Level 1 test dataset
2. Uses HuggingFace backend with meta-llama/Llama-3.2-1B
3. Uses default sampling config
4. Uses MathTask to format questions, generate responses, and evaluate rewards
5. Saves results to 00-sample/output/
"""

import json
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_onoff.backends import create_backend
from rl_onoff.backends.config import BackendConfig
from rl_onoff.sampling import Sampler
from rl_onoff.sampling.config import SamplingConfig
from rl_onoff.tasks import create_task
from rl_onoff.utils.dataset import GSM8KLevel1Dataset


def load_experiment_config(config_path: Optional[str] = None) -> dict:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: 00-sample_config.yaml in same directory)
        
    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = Path(__file__).parent / "00-sample_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main(
    experiment_config_path: Optional[str] = None
):
    """Main function to sample responses for GSM8K dataset.
    
    Args:
        experiment_config_path: Path to experiment config file (default: 00-sample_config.yaml)
    """
    # Load experiment configuration
    exp_config = load_experiment_config(experiment_config_path)
    
    # Set up output directory
    output_dir_str = exp_config.get("output", {}).get("dir", "exp/00-sample/output")
    output_dir = (project_root / output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GSM8K Sampling Experiment")
    print("=" * 80)
    
    # Load task config
    task_config_path = exp_config.get("task_config", "rl_onoff/tasks/configs/math_default.yaml")
    task_config = project_root / task_config_path
    print(f"\nLoading task config from {task_config}...")
    task = create_task(task_config)
    print(f"Task: template={task.config.template_type}, "
          f"format={task.config.format_type}, "
          f"reward={task.config.reward_type}")
    
    # Load dataset
    dataset_config = exp_config.get("dataset", {})
    dataset_name = dataset_config.get("name", "gsm8k_level1")
    dataset_split = dataset_config.get("split", "test")
    num_examples = dataset_config.get("num_examples")
    
    print(f"\nLoading {dataset_name} {dataset_split} dataset...")
    if dataset_name == "gsm8k_level1":
        dataset = GSM8KLevel1Dataset(split=dataset_split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset.load()
    print(f"Loaded {len(dataset)} examples")
    
    # Get all questions and answers
    questions = dataset.get_questions()
    answers = dataset.get_answers()
    
    if num_examples is not None:
        questions = questions[:num_examples]
        answers = answers[:num_examples]
        print(f"Processing first {num_examples} examples")
    
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
    
    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = Sampler(backend)
    
    # Load sampling config
    sampling_config_path_str = exp_config.get("sampling_config", "rl_onoff/sampling/configs/default.yaml")
    sampling_config_path = project_root / sampling_config_path_str
    print(f"\nLoading sampling config from {sampling_config_path}...")
    sampling_config = SamplingConfig.from_file(sampling_config_path)
    print(f"Sampling config: max_length={sampling_config.max_length}, "
          f"temperature={sampling_config.temperature}, "
          f"num_samples={sampling_config.num_samples}")
    
    # Format all questions into prompts
    print("\n" + "=" * 80)
    print("Formatting prompts...")
    print("=" * 80)
    prompts = []
    for question in tqdm(questions, desc="Formatting"):
        prompt = task.format_query(question)
        prompts.append(prompt)
    
    # Generate responses for all prompts at once
    print("\n" + "=" * 80)
    print("Generating responses...")
    print("=" * 80)
    print(f"Generating {sampling_config.num_samples} sample(s) per prompt for {len(prompts)} prompts...")
    all_responses = sampler.sample(prompts, config=sampling_config)
    
    # Process results
    print("\n" + "=" * 80)
    print("Evaluating responses...")
    print("=" * 80)
    
    all_results = []
    total_accuracy = 0.0
    total_length = 0
    num_correct = 0
    
    for i, (question, reference_answer, responses) in enumerate(tqdm(zip(questions, answers, all_responses), total=len(questions), desc="Evaluating")):
        # Handle multiple samples per prompt
        if sampling_config.num_samples > 1:
            # responses is a list of samples for this prompt
            response_list = responses
        else:
            # responses is a single string for this prompt
            response_list = [responses]
        
        # Evaluate each response
        sample_results = []
        for sample_idx, response in enumerate(response_list):
            # Get reward/score
            score = task.evaluate(response, reference_answer)
            
            # Extract answer for display
            extracted = task.extract_answer(response)
            extracted_answer = extracted.get("answer", "")
            
            # Track statistics
            if score > 0.5:  # Consider > 0.5 as correct
                num_correct += 1
            total_accuracy += score
            total_length += len(response)
            
            sample_result = {
                "sample_id": sample_idx,
                "response": response,
                "extracted_answer": extracted_answer,
                "reward": float(score),
                "response_length": len(response)
            }
            sample_results.append(sample_result)
        
        # Store result for this question
        result = {
            "example_id": i,
            "question": question,
            "reference_answer": reference_answer,
            "prompt": prompts[i],
            "samples": sample_results
        }
        all_results.append(result)
    
    # Calculate overall statistics
    total_samples = len(all_results) * sampling_config.num_samples
    overall_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
    accuracy_rate = num_correct / total_samples if total_samples > 0 else 0.0
    avg_length = total_length / total_samples if total_samples > 0 else 0.0
    
    statistics = {
        "dataset": f"{dataset_name} ({dataset_split})",
        "model": backend_config.model_name,
        "backend": backend_config.backend_type,
        "task_config_path": str(task_config),
        "num_examples": len(all_results),
        "num_samples_per_example": sampling_config.num_samples,
        "total_samples": total_samples,
        "overall_accuracy": overall_accuracy,
        "accuracy_rate": accuracy_rate,  # Percentage of samples with score > 0.5
        "average_response_length": avg_length,
        "sampling_config": {
            "max_length": sampling_config.max_length,
            "temperature": sampling_config.temperature,
            "top_k": sampling_config.top_k,
            "top_p": sampling_config.top_p,
            "do_sample": sampling_config.do_sample,
            "num_samples": sampling_config.num_samples,
            "batch_size": sampling_config.batch_size,
        },
        "task_config": {
            "template_type": task.config.template_type,
            "format_type": task.config.format_type,
            "reward_type": task.config.reward_type,
        }
    }
    
    # Save results
    results_file = output_dir / "results.json"
    print(f"\nSaving results to {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "statistics": statistics,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_results)} examples to {results_file}")
    
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
    print(f"Total examples: {len(all_results)}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Accuracy rate (>0.5): {accuracy_rate:.4f} ({num_correct}/{total_samples})")
    print(f"Average response length: {avg_length:.1f} characters")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample responses for GSM8K dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config file (default: 00-sample_config.yaml)"
    )
    
    args = parser.parse_args()
    main(experiment_config_path=args.config)

