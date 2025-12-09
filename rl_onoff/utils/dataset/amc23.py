"""AMC23 dataset."""

from pathlib import Path
from typing import Dict

from rl_onoff.utils.dataset.base import BaseDataset


class AMC23Dataset(BaseDataset):
    """AMC23 dataset loader.
    
    Format:
    - Question: prompt.content
    - Answer: reward_model.ground_truth.target
    """
    
    def get_file_path(self) -> Path:
        """Get the path to the AMC23 test file."""
        return self.data_dir / "data" / "amc23" / "test.parquet"
    
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

