"""Configuration management utilities."""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    """Configuration class for managing settings."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """Save config to JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary.
        
        Args:
            data: Dictionary of config values
            
        Returns:
            Config instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'Config':
        """Load config from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Config instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

