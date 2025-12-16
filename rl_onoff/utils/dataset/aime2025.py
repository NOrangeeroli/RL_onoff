"""AIME2025 dataset."""

from pathlib import Path
from typing import Dict

from rl_onoff.utils.dataset.base import BaseDataset


class AIME2025Dataset(BaseDataset):
    """AIME2025 dataset loader.
    
    Format:
    - Question: prompt.content
    - Answer: reward_model.ground_truth.target
    """
    
    def get_file_path(self) -> Path:
        """Get the path to the AIME2025 test file."""
        return self.data_dir / "data" / "aime2025" / "test.parquet"
    
    def extract_question(self, entry: Dict) -> str:
        """Extract question from entry."""
        prompt = entry.get("prompt", {})
        
        # Handle case where prompt is a list of length 1 containing a dict
        if isinstance(prompt, list) and len(prompt) > 0:
            prompt = prompt[0]
        
        if isinstance(prompt, dict):
            content = prompt.get("content", "")
            # Ensure it's a string
            return str(content) if content else ""
        # If prompt is not a dict, convert to string
        return str(prompt) if prompt else ""
    
    def extract_answer(self, entry: Dict) -> str:
        """Extract answer from entry."""
        reward_model = entry.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", {})
            if isinstance(ground_truth, dict):
                return str(ground_truth.get("target", ""))
            return str(ground_truth)
        return ""


if __name__ == "__main__":
    """Load and print one entry from AIME2025 dataset."""
    try:
        dataset = AIME2025Dataset()
        question, answer, solution = dataset[0]
        
        print("=" * 60)
        print("AIME2025 Dataset - First Entry")
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Solution: {solution}")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
    except Exception as e:
        print(f"Error loading dataset: {e}")

