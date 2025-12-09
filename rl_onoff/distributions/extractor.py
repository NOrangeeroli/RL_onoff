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
        
        # Get logits/probabilities for all solution tokens at once
        if use_logits:
            # Get logits for all solution tokens
            distribution_array = self.backend.get_logits(question, solution)
        else:
            # Get probabilities for all solution tokens
            distribution_array = self.backend.get_probabilities(
                question, solution, temperature=temperature
            )
        
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


if __name__ == "__main__":
    """Simple use cases for DistributionExtractor."""
    
    try:
        from rl_onoff.backends import HuggingFaceBackend
        
        print("Running DistributionExtractor examples...")
        print("=" * 60)
        
        # Initialize backend (using a small model for demonstration)
        from rl_onoff.backends.config import BackendConfig
        from rl_onoff.backends import create_backend
        model_name = "gpt2"  # Replace with your preferred model
        config = BackendConfig(backend_type="huggingface", model_name=model_name)
        backend = create_backend(config)
        
        # Create distribution extractor
        extractor = DistributionExtractor(backend)
        print(f"DistributionExtractor initialized with backend: {model_name}\n")
        
        # Example 1: Extract probabilities for a single question-solution pair
        print("=" * 60)
        print("Example 1: Extract probabilities for a single question-solution pair")
        print("=" * 60)
        question = "What is 2 + 2?"
        solution = " The answer is 4."
        probs = extractor.extract_distributions(
            question=question,
            solution=solution,
            use_logits=False,
            temperature=1.0
        )
        print(f"Question: {question}")
        print(f"Solution: {solution}")
        print(f"Probabilities shape: {probs.shape}")
        print(f"Number of tokens in solution: {probs.shape[0]}")
        print(f"Vocabulary size: {probs.shape[1]}")
        print(f"First token probabilities (top 5): {np.argsort(probs[0])[-5:][::-1]}\n")
        
        # Example 2: Extract logits instead of probabilities
        print("=" * 60)
        print("Example 2: Extract logits instead of probabilities")
        print("=" * 60)
        logits = extractor.extract_distributions(
            question=question,
            solution=solution,
            use_logits=True
        )
        print(f"Logits shape: {logits.shape}")
        print(f"First token logits (top 5): {np.argsort(logits[0])[-5:][::-1]}\n")
        
        # Example 3: Extract with token IDs
        print("=" * 60)
        print("Example 3: Extract distributions with token IDs")
        print("=" * 60)
        probs_with_ids, token_ids = extractor.extract_distributions(
            question=question,
            solution=solution,
            return_token_ids=True
        )
        print(f"Probabilities shape: {probs_with_ids.shape}")
        print(f"Token IDs: {token_ids}")
        print(f"Number of tokens: {len(token_ids)}\n")
        
        # Example 4: Batch extraction
        print("=" * 60)
        print("Example 4: Batch extraction for multiple question-solution pairs")
        print("=" * 60)
        questions = [
            "What is 2 + 2?",
            "What is the capital of France?"
        ]
        solutions = [
            " The answer is 4.",
            " The capital is Paris."
        ]
        batch_results = extractor.extract_distributions_batch(
            questions=questions,
            solutions=solutions,
            use_logits=False
        )
        print(f"Number of results: {len(batch_results)}")
        for i, (q, s, result) in enumerate(zip(questions, solutions, batch_results)):
            print(f"  Pair {i+1}:")
            print(f"    Question: {q}")
            print(f"    Solution: {s}")
            print(f"    Distribution shape: {result.shape}\n")
        
        # Example 5: Get vocabulary size
        print("=" * 60)
        print("Example 5: Get vocabulary size")
        print("=" * 60)
        vocab_size = extractor.get_vocab_size()
        print(f"Vocabulary size: {vocab_size}\n")
        
        # Example 6: Token ID conversions
        print("=" * 60)
        print("Example 6: Token ID conversions")
        print("=" * 60)
        test_token = "hello"
        token_id = extractor.get_token_to_id(test_token)
        if token_id is not None:
            print(f"Token '{test_token}' -> ID: {token_id}")
            converted_back = extractor.get_id_to_token(token_id)
            print(f"ID {token_id} -> Token: '{converted_back}'\n")
        else:
            print(f"Could not convert token '{test_token}' to ID\n")
        
        # Example 7: Extract with different temperature
        print("=" * 60)
        print("Example 7: Extract with different temperature")
        print("=" * 60)
        probs_temp_high = extractor.extract_distributions(
            question=question,
            solution=solution,
            temperature=2.0  # Higher temperature = more uniform distribution
        )
        probs_temp_low = extractor.extract_distributions(
            question=question,
            solution=solution,
            temperature=0.5  # Lower temperature = sharper distribution
        )
        print(f"High temperature (2.0) - entropy: {np.sum(-probs_temp_high[0] * np.log(probs_temp_high[0] + 1e-10)):.4f}")
        print(f"Low temperature (0.5) - entropy: {np.sum(-probs_temp_low[0] * np.log(probs_temp_low[0] + 1e-10)):.4f}\n")
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure HuggingFace backend is available.")
    except NotImplementedError as e:
        print(f"\nNote: {e}")
        print("This backend does not support logit extraction.")
        print("Please use HuggingFace backend for distribution extraction.")
    except Exception as e:
        print(f"\nAn error occurred during DistributionExtractor examples: {e}")
        print("Please ensure you have a compatible model and sufficient resources.")

