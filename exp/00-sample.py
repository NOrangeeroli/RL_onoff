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
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_onoff.backends import create_backend
from rl_onoff.backends.config import BackendConfig
from rl_onoff.sampling import Sampler
from rl_onoff.sampling.config import SamplingConfig
from rl_onoff.tasks.math import MathTask
from rl_onoff.utils.dataset import GSM8KLevel1Dataset


def main():
    """Main function to sample responses for GSM8K dataset."""
    # Set up output directory
    output_dir = Path(__file__).parent / "00-sample" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GSM8K Sampling Experiment")
    print("=" * 80)
    
    # Load GSM8K dataset
    print("\nLoading GSM8K Level 1 test dataset...")
    dataset = GSM8KLevel1Dataset(split="test")
    dataset.load()
    print(f"Loaded {len(dataset)} examples")
    
    # Get all questions and answers
    questions = dataset.get_questions()[:10]
    answers = dataset.get_answers()[:10]
    
    # Initialize backend
    print("\nInitializing HuggingFace backend...")
    backend_config = BackendConfig(
        backend_type="huggingface",
        model_name="Qwen/Qwen3-4B"
    )
    backend = create_backend(backend_config)
    backend.load()
    print("Backend loaded successfully!")
    
    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = Sampler(backend)
    
    # Load default sampling config
    print("\nLoading default sampling config...")
    sampling_config_path = project_root / "rl_onoff" / "sampling" / "configs" / "default.yaml"
    sampling_config = SamplingConfig.from_file(sampling_config_path)
    print(f"Sampling config: max_length={sampling_config.max_length}, "
          f"temperature={sampling_config.temperature}, "
          f"num_samples={sampling_config.num_samples}")
    
    # Initialize MathTask
    print("\nInitializing MathTask...")
    task = MathTask()
    print(f"Task: template={task.config.template_type}, "
          f"format={task.config.format_type}, "
          f"reward={task.config.reward_type}")
    
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
        "dataset": "GSM8K Level 1 (test)",
        "model": "meta-llama/Llama-3.2-1B",
        "backend": "huggingface",
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
    main()

