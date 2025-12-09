"""MATH dataset."""

from pathlib import Path
from typing import Dict

from rl_onoff.utils.dataset.base import BaseDataset


class MathDataset(BaseDataset):
    """MATH dataset loader.
    
    Format:
    - Question: prompt.content
    - Answer: reward_model.ground_truth.target
    """
    
    def get_file_path(self) -> Path:
        """Get the path to the MATH file."""
        return self.data_dir / "data" / "math" / f"{self.split}.parquet"
    
    def extract_question(self, entry: Dict) -> str:
        """Extract question from entry."""
        prompt = entry.get("prompt", {})
        if isinstance(prompt, dict):
            return prompt.get("content", "")
        return str(prompt)
    
    def extract_answer(self, entry: Dict) -> str:
        """Extract answer from entry."""
        reward_model = entry.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", {})
            if isinstance(ground_truth, dict):
                return str(ground_truth.get("target", ""))
            return str(ground_truth)
        return ""

