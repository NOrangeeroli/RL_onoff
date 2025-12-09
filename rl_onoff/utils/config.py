"""Configuration management utilities."""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


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
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save config to YAML file.
        
        Args:
            file_path: Path to save YAML file
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is not installed. Install it with: pip install pyyaml"
            )
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
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
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'Config':
        """Load config from YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Config instance
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is not installed. Install it with: pip install pyyaml"
            )
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """Load config from file (auto-detect JSON or YAML).
        
        Args:
            file_path: Path to config file
            
        Returns:
            Config instance
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.yaml' or suffix == '.yml':
            return cls.from_yaml(file_path)
        elif suffix == '.json':
            return cls.from_json(file_path)
        else:
            # Try YAML first, then JSON
            try:
                return cls.from_yaml(file_path)
            except Exception:
                return cls.from_json(file_path)

