"""Conditional model switching during generation."""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np

from rl_onoff.backends.base import BaseBackend
from rl_onoff.divergence import DivergenceCalculator


class ModelSwitcher:
    """Switch between models during generation based on divergence thresholds."""

    def __init__(
        self,
        backend_a: BaseBackend,
        backend_b: BaseBackend,
        divergence_type: str = "js",
        threshold: float = 0.5,
        switch_back_threshold: Optional[float] = None,
    ):
        """Initialize model switcher.
        
        Args:
            backend_a: Primary model (start with this)
            backend_b: Secondary model (switch to this when threshold exceeded)
            divergence_type: Type of divergence to use ("kl" or "js")
            threshold: Divergence threshold for switching from A to B
            switch_back_threshold: Divergence threshold for switching back to A
                (if None, uses threshold / 2)
        """
        self.backend_a = backend_a
        self.backend_b = backend_b
        self.divergence_type = divergence_type
        self.threshold = threshold
        self.switch_back_threshold = switch_back_threshold or (threshold / 2.0)
        
        self.divergence_calculator = DivergenceCalculator()
        
        # Ensure backends are loaded
        if not backend_a.is_loaded():
            backend_a.load()
        if not backend_b.is_loaded():
            backend_b.load()

    def generate_with_switching(
        self,
        question: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        return_switch_points: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Generate response with conditional model switching.
        
        Args:
            question: Input question/prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            return_switch_points: Whether to return switch point information
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text, optionally with switch points info
            Switch points is a list of dicts with keys:
            - "position": token position where switch occurred
            - "from_model": model being switched from ("a" or "b")
            - "to_model": model being switched to ("a" or "b")
            - "divergence": divergence value at switch point
        """
        tokenizer_a = self.backend_a.get_tokenizer()
        current_model = "a"
        current_backend = self.backend_a
        
        generated_tokens = []
        switch_points = []
        current_text = question
        
        for step in range(max_new_tokens):
            # Get distributions from both models at current position
            try:
                # Get probabilities from both models
                probs_a = self.backend_a.get_probabilities(
                    current_text, max_new_tokens=1, temperature=1.0
                )
                probs_b = self.backend_b.get_probabilities(
                    current_text, max_new_tokens=1, temperature=1.0
                )
                
                # Handle different return types (single array or list)
                if isinstance(probs_a, list):
                    dist_a = probs_a[0]  # (vocab_size,)
                else:
                    dist_a = probs_a[0] if len(probs_a.shape) > 1 else probs_a  # (vocab_size,)
                
                if isinstance(probs_b, list):
                    dist_b = probs_b[0]  # (vocab_size,)
                else:
                    dist_b = probs_b[0] if len(probs_b.shape) > 1 else probs_b  # (vocab_size,)
                
                # Ensure correct shape
                if len(dist_a.shape) == 1:
                    dist_a = dist_a.reshape(1, -1)  # (1, vocab_size)
                if len(dist_b.shape) == 1:
                    dist_b = dist_b.reshape(1, -1)  # (1, vocab_size)
                
                # Compute divergence at this position
                divergences = self.divergence_calculator.compute_token_divergences(
                    dist_a,
                    dist_b,
                    divergence_type=self.divergence_type,
                )
                divergence_value = divergences[self.divergence_type][0]
                
                # Decide whether to switch
                if current_model == "a":
                    if divergence_value > self.threshold:
                        # Switch to model B
                        current_model = "b"
                        current_backend = self.backend_b
                        switch_points.append({
                            "position": step,
                            "from_model": "a",
                            "to_model": "b",
                            "divergence": float(divergence_value),
                        })
                else:  # current_model == "b"
                    if divergence_value < self.switch_back_threshold:
                        # Switch back to model A
                        current_model = "a"
                        current_backend = self.backend_a
                        switch_points.append({
                            "position": step,
                            "from_model": "b",
                            "to_model": "a",
                            "divergence": float(divergence_value),
                        })
                
                # Generate next token using current model
                next_token_text = current_backend.generate(
                    current_text,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    **kwargs
                )
                
                # Extract just the new token (remove the prompt part)
                new_token = next_token_text[len(current_text):]
                generated_tokens.append(new_token)
                current_text = current_text + new_token
                
            except Exception as e:
                # If something goes wrong, continue with current model
                # or fall back gracefully
                print(f"Warning: Error during generation step {step}: {e}")
                next_token_text = current_backend.generate(
                    current_text,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    **kwargs
                )
                new_token = next_token_text[len(current_text):]
                generated_tokens.append(new_token)
                current_text = current_text + new_token
        
        generated_text = "".join(generated_tokens)
        
        if return_switch_points:
            return generated_text, switch_points
        return generated_text

    def generate_with_switching_batch(
        self,
        questions: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        return_switch_points: bool = False,
        **kwargs
    ) -> Union[List[str], List[Tuple[str, List[Dict]]]]:
        """Generate responses for multiple questions with switching.
        
        Args:
            questions: List of input questions
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            return_switch_points: Whether to return switch point information
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated texts, optionally with switch points
        """
        results = []
        
        for question in questions:
            result = self.generate_with_switching(
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                return_switch_points=return_switch_points,
                **kwargs
            )
            results.append(result)
        
        return results

