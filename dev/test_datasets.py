#!/usr/bin/env python3
"""Test dataset classes."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rl_onoff.utils.dataset import (
    AIME2025Dataset,
    AMC23Dataset,
    GSM8KLevel1Dataset,
    MathDataset,
)


def test_dataset(dataset_class, dataset_name, split="test"):
    """Test a dataset class."""
    print("=" * 80)
    print(f"Testing {dataset_name} ({split})")
    print("=" * 80)
    
    try:
        dataset = dataset_class(split=split)
        print(f"Dataset file: {dataset.get_file_path()}")
        print(f"File exists: {dataset.get_file_path().exists()}")
        
        if not dataset.get_file_path().exists():
            print("⚠️  File not found, skipping...\n")
            return
        
        # Load and test
        dataset.load()
        print(f"Loaded {len(dataset)} examples")
        
        # Get first example
        if len(dataset) > 0:
            question, answer = dataset[0]
            print(f"\nFirst example:")
            print(f"  Question: {question[:100]}..." if len(question) > 100 else f"  Question: {question}")
            print(f"  Answer: {answer}")
        
        # Test get_all
        all_pairs = dataset.get_all()
        print(f"\nTotal pairs: {len(all_pairs)}")
        
        # Test get_questions and get_answers
        questions = dataset.get_questions()
        answers = dataset.get_answers()
        print(f"Questions: {len(questions)}")
        print(f"Answers: {len(answers)}")
        
        print("✅ Test passed!\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    print("Dataset Classes Test")
    print("=" * 80)
    print()
    
    # Test each dataset
    test_dataset(AIME2025Dataset, "AIME2025", split="test")
    test_dataset(AMC23Dataset, "AMC23", split="test")
    test_dataset(GSM8KLevel1Dataset, "GSM8K Level 1", split="test")
    test_dataset(GSM8KLevel1Dataset, "GSM8K Level 1", split="train")
    test_dataset(MathDataset, "MATH", split="test")
    test_dataset(MathDataset, "MATH", split="train")
    
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)

