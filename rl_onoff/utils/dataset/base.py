"""Base dataset class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from rl_onoff.utils.data_loader import load_data


class BaseDataset(ABC):
    """Base class for dataset loaders.
    
    Each dataset class should:
    1. Know the file path structure
    2. Load data using data_loader
    3. Extract question, answer, and solution strings
    
    The dataset returns tuples of (question, answer, solution) where solution
    may be None if not available in the data.
    """
    
    def __init__(self, data_dir: Optional[Path] = None, split: str = "test"):
        """Initialize dataset.
        
        Args:
            data_dir: Root directory containing data folder (default: project root)
            split: Dataset split to load ("train" or "test")
        """
        if data_dir is None:
            # Default to project root (assuming data/ is at project root)
            data_dir = Path(__file__).parent.parent.parent.parent
        
        self.data_dir = Path(data_dir)
        self.split = split
        self._data: Optional[List[Dict]] = None
    
    @abstractmethod
    def get_file_path(self) -> Path:
        """Get the path to the data file for this dataset.
        
        Returns:
            Path to the data file
        """
        pass
    
    @abstractmethod
    def extract_question(self, entry: Dict) -> str:
        """Extract question string from a data entry.
        
        Args:
            entry: Dictionary entry from loaded data
            
        Returns:
            Question string
        """
        pass
    
    @abstractmethod
    def extract_answer(self, entry: Dict) -> str:
        """Extract answer string from a data entry.
        
        Args:
            entry: Dictionary entry from loaded data
            
        Returns:
            Answer string
        """
        pass
    
    def extract_solution(self, entry: Dict) -> Optional[str]:
        """Extract solution string from a data entry.
        
        Tries to get solution from reward_model.ground_truth.solution.
        Returns None if not found.
        
        Args:
            entry: Dictionary entry from loaded data
            
        Returns:
            Solution string or None if not available
        """
        reward_model = entry.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", {})
            if isinstance(ground_truth, dict):
                solution = ground_truth.get("solution")
                if solution is not None:
                    return str(solution)
        return None
    
    def load(self) -> None:
        """Load the dataset from file."""
        file_path = self.get_file_path()
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        self._data = load_data(file_path)
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        if self._data is None:
            self.load()
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, Optional[str]]:
        """Get a question-answer-solution triple by index.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tuple of (question, answer, solution) where solution may be None
        """
        if self._data is None:
            self.load()
        
        entry = self._data[idx]
        question = self.extract_question(entry)
        answer = self.extract_answer(entry)
        solution = self.extract_solution(entry)
        # Ensure question and answer are strings
        return str(question), str(answer), solution
    
    def get_all(self) -> List[Tuple[str, str, Optional[str]]]:
        """Get all question-answer-solution triples.
        
        Returns:
            List of (question, answer, solution) tuples where solution may be None
        """
        if self._data is None:
            self.load()
        
        return [self.__getitem__(i) for i in range(len(self._data))]
    
    def get_questions(self) -> List[str]:
        """Get all questions.
        
        Returns:
            List of question strings
        """
        if self._data is None:
            self.load()
        
        return [str(self.extract_question(entry)) for entry in self._data]
    
    def get_answers(self) -> List[str]:
        """Get all answers.
        
        Returns:
            List of answer strings
        """
        if self._data is None:
            self.load()
        
        return [str(self.extract_answer(entry)) for entry in self._data]
    
    def get_solutions(self) -> List[Optional[str]]:
        """Get all solutions.
        
        Returns:
            List of solution strings (may contain None if not available)
        """
        if self._data is None:
            self.load()
        
        return [self.extract_solution(entry) for entry in self._data]

