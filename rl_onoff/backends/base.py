"""Base backend interface for LLM inference engines."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import numpy as np


class BaseBackend(ABC):
    """Abstract base class for all model backends."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize the backend.
        
        Args:
            model_name: Name or path of the model to load
            **kwargs: Additional backend-specific arguments
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load the model and tokenizer.
        
        Args:
            **kwargs: Additional loading arguments
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        stop_strings: Optional[List[str]] = None,
        return_logits: bool = False,
        return_probs: bool = False,
        **kwargs
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Generate text from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            stop_strings: List of strings that will stop generation when encountered
            return_logits: If True, return logits along with generated text
            return_probs: If True, return probabilities along with generated text
            **kwargs: Additional generation arguments
            
        Returns:
            If return_logits=False and return_probs=False:
                Generated text(s) as string or list of strings
            If return_logits=True or return_probs=True:
                Dictionary or list of dictionaries with keys:
                - 'text': Generated text
                - 'logits': Logits array (if return_logits=True)
                - 'probs': Probabilities array (if return_probs=True)
        """
        pass

    @abstractmethod
    def get_logits(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token logits for predicting response tokens given prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses to get logits for
            **kwargs: Additional arguments
            
        Returns:
            Logits array(s) with shape (batch_size, response_len, vocab_size) or
            list of arrays with shape (response_len, vocab_size)
            where response_len is the number of tokens in the response
        """
        pass

    def get_probabilities(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        temperature: float = 1.0,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get token probability distributions for predicting response tokens given prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            responses: Single response or list of responses to get probabilities for
            temperature: Temperature for softmax normalization
            **kwargs: Additional arguments
            
        Returns:
            Probability array(s) with shape (batch_size, response_len, vocab_size) or
            list of arrays with shape (response_len, vocab_size)
            where response_len is the number of tokens in the response
        """
        logits = self.get_logits(prompts, responses, **kwargs)
        
        # Apply temperature and softmax
        if isinstance(logits, list):
            probs = []
            for logit_array in logits:
                scaled_logits = logit_array / temperature
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
                prob_array = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                probs.append(prob_array)
            return probs
        else:
            scaled_logits = logits / temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return probs

    @abstractmethod
    def get_tokenizer(self):
        """Get the tokenizer instance.
        
        Returns:
            Tokenizer instance
        """
        pass

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs.
        
        Args:
            text: Text or list of texts to encode
            
        Returns:
            Token IDs or list of token ID lists
        """
        tokenizer = self.get_tokenizer()
        if isinstance(text, str):
            return tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.encode(t, add_special_tokens=False) for t in text]

    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """Decode token IDs to text.
        
        Args:
            token_ids: Token IDs or list of token ID lists
            
        Returns:
            Decoded text or list of texts
        """
        tokenizer = self.get_tokenizer()
        if isinstance(token_ids[0], int):
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]

    def is_loaded(self) -> bool:
        """Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._is_loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name})"

