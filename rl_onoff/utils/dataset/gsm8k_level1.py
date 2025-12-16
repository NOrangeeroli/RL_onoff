"""GSM8K Level 1 dataset."""

from pathlib import Path
from typing import Dict

from rl_onoff.utils.dataset.base import BaseDataset


class GSM8KLevel1Dataset(BaseDataset):
    """GSM8K Level 1 dataset loader.
    
    Format:
    - Question: extra_info.question or prompt.content
    - Answer: reward_model.ground_truth (string) or extra_info.answer
    """
    
    def get_file_path(self) -> Path:
        """Get the path to the GSM8K Level 1 file."""
        return self.data_dir / "data" / "gsm8k_level1" / f"{self.split}.parquet"
    
    def extract_question(self, entry: Dict) -> str:
        """Extract question from entry."""
        # Try extra_info.question first, then prompt.content
        extra_info = entry.get("extra_info", {})
        if isinstance(extra_info, dict):
            question = extra_info.get("question")
            if question:
                return str(question) if question else ""
        
        prompt = entry.get("prompt", {})
        if isinstance(prompt, dict):
            content = prompt.get("content", "")
            # Ensure it's a string
            return str(content) if content else ""
        # If prompt is not a dict, convert to string
        return str(prompt) if prompt else ""
    
    def extract_answer(self, entry: Dict) -> str:
        """Extract answer from entry."""
        # Try extra_info.answer first, then reward_model.ground_truth
        extra_info = entry.get("extra_info", {})
        if isinstance(extra_info, dict):
            answer = extra_info.get("answer")
            if answer:
                return str(answer)
        
        reward_model = entry.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
            if ground_truth:
                return str(ground_truth)
        return ""


if __name__ == "__main__":
    """Load and print one entry from GSM8K Level 1 dataset."""
    try:
        dataset = GSM8KLevel1Dataset(split="test")
        question, answer, solution = dataset[0]
        
        print("=" * 60)
        print("GSM8K Level 1 Dataset - First Entry")
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Solution: {solution}")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
    except Exception as e:
        print(f"Error loading dataset: {e}")

