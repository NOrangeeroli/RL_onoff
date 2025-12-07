"""Token-wise distribution extraction from models."""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np

from rl_onoff.backends.base import BaseBackend


class DistributionExtractor:
    """Extract token-wise probability distributions from models."""

    def __init__(self, backend: BaseBackend):
        """Initialize distribution extractor.
        
        Args:
            backend: Model backend instance
        """
        self.backend = backend
        if not backend.is_loaded():
            backend.load()

    def extract_distributions(
        self,
        question: str,
        solution: str,
        use_logits: bool = False,
        temperature: float = 1.0,
        return_token_ids: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
        """Extract token-wise distributions for a solution given a question.
        
        Args:
            question: Input question/prompt
            solution: Solution text to extract distributions for
            use_logits: If True, return logits; if False, return probabilities
            temperature: Temperature for probability normalization (if use_logits=False)
            return_token_ids: Whether to also return token IDs
            
        Returns:
            If use_logits=True: logits array with shape (seq_len, vocab_size)
            If use_logits=False: probabilities array with shape (seq_len, vocab_size)
            If return_token_ids=True: tuple of (distribution_array, token_ids)
        """
        tokenizer = self.backend.get_tokenizer()
        
        # Tokenize question and solution
        question_ids = self.backend.encode(question)
        solution_ids = self.backend.encode(solution)
        
        # Create full prompt (question + solution)
        full_prompt = question + solution
        full_prompt_ids = self.backend.encode(full_prompt)
        
        # Extract distributions token by token
        distributions = []
        current_prompt = question
        
        for i, solution_token_id in enumerate(solution_ids):
            # Get logits for the next token position
            logits = self.backend.get_logits(current_prompt, max_new_tokens=1)
            
            # logits shape: (1, vocab_size) - take the first (and only) token
            next_token_logits = logits[0]  # (vocab_size,)
            
            if use_logits:
                distributions.append(next_token_logits)
            else:
                # Convert to probabilities
                probs = self.backend.get_probabilities(
                    current_prompt, max_new_tokens=1, temperature=temperature
                )
                next_token_probs = probs[0]  # (vocab_size,)
                distributions.append(next_token_probs)
            
            # Update prompt with the next token
            solution_token = tokenizer.decode([solution_token_id], skip_special_tokens=True)
            current_prompt = current_prompt + solution_token
        
        # Stack into array: (seq_len, vocab_size)
        distribution_array = np.stack(distributions)
        
        if return_token_ids:
            return distribution_array, solution_ids
        return distribution_array

    def extract_distributions_batch(
        self,
        questions: List[str],
        solutions: List[str],
        use_logits: bool = False,
        temperature: float = 1.0,
        return_token_ids: bool = False,
    ) -> Union[List[np.ndarray], List[Tuple[np.ndarray, List[int]]]]:
        """Extract distributions for multiple question-solution pairs.
        
        Args:
            questions: List of input questions/prompts
            solutions: List of solution texts
            use_logits: If True, return logits; if False, return probabilities
            temperature: Temperature for probability normalization
            return_token_ids: Whether to also return token IDs
            
        Returns:
            List of distribution arrays or list of tuples (distribution_array, token_ids)
        """
        results = []
        
        for question, solution in zip(questions, solutions):
            result = self.extract_distributions(
                question=question,
                solution=solution,
                use_logits=use_logits,
                temperature=temperature,
                return_token_ids=return_token_ids,
            )
            results.append(result)
        
        return results

    def get_vocab_size(self) -> int:
        """Get vocabulary size of the model.
        
        Returns:
            Vocabulary size
        """
        tokenizer = self.backend.get_tokenizer()
        if hasattr(tokenizer, 'vocab_size'):
            return tokenizer.vocab_size
        elif hasattr(tokenizer, 'vocab'):
            return len(tokenizer.vocab)
        else:
            # Fallback: try to get from model config
            if hasattr(self.backend, 'model') and hasattr(self.backend.model, 'config'):
                return self.backend.model.config.vocab_size
            else:
                raise ValueError("Could not determine vocabulary size")

    def get_token_to_id(self, token: str) -> Optional[int]:
        """Get token ID for a given token string.
        
        Args:
            token: Token string
            
        Returns:
            Token ID or None if not found
        """
        tokenizer = self.backend.get_tokenizer()
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            return tokenizer.convert_tokens_to_ids(token)
        elif hasattr(tokenizer, 'encode'):
            ids = tokenizer.encode(token, add_special_tokens=False)
            return ids[0] if len(ids) > 0 else None
        else:
            return None

    def get_id_to_token(self, token_id: int) -> Optional[str]:
        """Get token string for a given token ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string or None if not found
        """
        tokenizer = self.backend.get_tokenizer()
        if hasattr(tokenizer, 'convert_ids_to_tokens'):
            return tokenizer.convert_ids_to_tokens(token_id)
        elif hasattr(tokenizer, 'decode'):
            return tokenizer.decode([token_id], skip_special_tokens=True)
        else:
            return None

