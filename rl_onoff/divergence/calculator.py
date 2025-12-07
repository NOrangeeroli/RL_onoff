"""Token-level divergence calculations between model distributions."""

from typing import List, Optional, Union, Dict
import numpy as np
from scipy.stats import entropy


class DivergenceCalculator:
    """Calculate token-level divergence metrics between distributions."""

    def __init__(self, epsilon: float = 1e-10):
        """Initialize divergence calculator.
        
        Args:
            epsilon: Small value to avoid numerical issues (e.g., log(0))
        """
        self.epsilon = epsilon

    def kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        """Compute KL divergence KL(P||Q) = sum(P * log(P / Q)).
        
        Args:
            p: First distribution array (can be 1D or multi-dimensional)
            q: Second distribution array (same shape as p)
            axis: Axis along which to compute divergence
            
        Returns:
            KL divergence values (same shape as p but with specified axis removed)
        """
        # Ensure distributions are normalized
        p = p / (np.sum(p, axis=axis, keepdims=True) + self.epsilon)
        q = q / (np.sum(q, axis=axis, keepdims=True) + self.epsilon)
        
        # Add epsilon to avoid log(0)
        p_safe = p + self.epsilon
        q_safe = q + self.epsilon
        
        # Compute KL divergence: sum(P * log(P / Q))
        kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=axis)
        
        return kl

    def js_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        axis: int = -1,
    ) -> np.ndarray:
        """Compute Jensen-Shannon divergence JS(P||Q).
        
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)
        
        Args:
            p: First distribution array (can be 1D or multi-dimensional)
            q: Second distribution array (same shape as p)
            axis: Axis along which to compute divergence
            
        Returns:
            JS divergence values (same shape as p but with specified axis removed)
        """
        # Ensure distributions are normalized
        p = p / (np.sum(p, axis=axis, keepdims=True) + self.epsilon)
        q = q / (np.sum(q, axis=axis, keepdims=True) + self.epsilon)
        
        # Compute mixture distribution M = 0.5 * (P + Q)
        m = 0.5 * (p + q)
        
        # Compute JS divergence
        kl_pm = self.kl_divergence(p, m, axis=axis)
        kl_qm = self.kl_divergence(q, m, axis=axis)
        js = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js

    def compute_token_divergences(
        self,
        dist_a: np.ndarray,
        dist_b: np.ndarray,
        divergence_type: str = "both",
    ) -> Dict[str, np.ndarray]:
        """Compute token-level divergences between two distributions.
        
        Args:
            dist_a: Distribution from model A, shape (seq_len, vocab_size)
            dist_b: Distribution from model B, shape (seq_len, vocab_size)
            divergence_type: Type of divergence to compute ("kl", "js", or "both")
            
        Returns:
            Dictionary with divergence arrays:
            - "kl": KL divergence at each token position (seq_len,)
            - "js": JS divergence at each token position (seq_len,)
        """
        if dist_a.shape != dist_b.shape:
            raise ValueError(
                f"Distribution shapes must match: {dist_a.shape} != {dist_b.shape}"
            )
        
        results = {}
        
        if divergence_type in ["kl", "both"]:
            kl = self.kl_divergence(dist_a, dist_b, axis=-1)
            results["kl"] = kl
        
        if divergence_type in ["js", "both"]:
            js = self.js_divergence(dist_a, dist_b, axis=-1)
            results["js"] = js
        
        return results

    def compute_divergence_for_solutions(
        self,
        question: str,
        solution: str,
        backend_a,
        backend_b,
        divergence_type: str = "both",
        use_logits: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Compute token-level divergences for a question-solution pair.
        
        Args:
            question: Input question/prompt
            solution: Solution text
            backend_a: First model backend
            backend_b: Second model backend
            divergence_type: Type of divergence ("kl", "js", or "both")
            use_logits: Whether to use logits instead of probabilities
            temperature: Temperature for probability normalization
            
        Returns:
            Dictionary with divergence arrays at each token position
        """
        from rl_onoff.distributions import DistributionExtractor
        
        # Extract distributions from both models
        extractor_a = DistributionExtractor(backend_a)
        extractor_b = DistributionExtractor(backend_b)
        
        dist_a = extractor_a.extract_distributions(
            question=question,
            solution=solution,
            use_logits=use_logits,
            temperature=temperature,
        )
        
        dist_b = extractor_b.extract_distributions(
            question=question,
            solution=solution,
            use_logits=use_logits,
            temperature=temperature,
        )
        
        # Compute divergences
        return self.compute_token_divergences(
            dist_a=dist_a,
            dist_b=dist_b,
            divergence_type=divergence_type,
        )

    def compute_batch_divergences(
        self,
        questions: List[str],
        solutions: List[str],
        backend_a,
        backend_b,
        divergence_type: str = "both",
        use_logits: bool = False,
        temperature: float = 1.0,
    ) -> List[Dict[str, np.ndarray]]:
        """Compute divergences for multiple question-solution pairs.
        
        Args:
            questions: List of input questions
            solutions: List of solution texts
            backend_a: First model backend
            backend_b: Second model backend
            divergence_type: Type of divergence ("kl", "js", or "both")
            use_logits: Whether to use logits instead of probabilities
            temperature: Temperature for probability normalization
            
        Returns:
            List of divergence dictionaries, one per question-solution pair
        """
        results = []
        
        for question, solution in zip(questions, solutions):
            result = self.compute_divergence_for_solutions(
                question=question,
                solution=solution,
                backend_a=backend_a,
                backend_b=backend_b,
                divergence_type=divergence_type,
                use_logits=use_logits,
                temperature=temperature,
            )
            results.append(result)
        
        return results


