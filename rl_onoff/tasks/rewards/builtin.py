"""Built-in rewards for evaluating model outputs."""

from typing import Union, List, Dict, Optional
import numpy as np

from rl_onoff.tasks.rewards.base import BaseReward

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    BLEU_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        BLEU_AVAILABLE = False
except ImportError:
    BLEU_AVAILABLE = False

try:
    from math_verify import verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


class PerplexityReward(BaseReward):
    """Compute perplexity of generated text."""

    def __init__(self, backend=None):
        """Initialize perplexity reward.
        
        Args:
            backend: Backend instance for computing log probabilities
        """
        super().__init__("perplexity")
        self.backend = backend

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Union[float, List[float]]:
        """Compute perplexity.
        
        Args:
            predictions: Generated text(s)
            references: Optional reference text(s) (not used for perplexity)
            **kwargs: Additional arguments
            
        Returns:
            Perplexity value(s)
        """
        if self.backend is None:
            raise ValueError("Backend required for perplexity computation")
        
        is_single = isinstance(predictions, str)
        if is_single:
            predictions = [predictions]

        perplexities = []
        for pred in predictions:
            # Tokenize and compute log probabilities
            token_ids = self.backend.encode(pred)
            if len(token_ids) == 0:
                perplexities.append(float('inf'))
                continue
            
            # Get probabilities for all tokens in the prediction
            # Use empty prompt and prediction as solution to get logits for each token
            # Note: logits at position i predict token at position i+1 in the full sequence
            # So for token at position i in the solution, we use logits at position i-1
            probs = self.backend.get_probabilities("", pred, temperature=1.0)
            
            # Compute perplexity: exp(-1/N * sum(log P(token_i | tokens_0...i-1)))
            log_probs = np.log(probs + 1e-10)  # Add small epsilon for numerical stability
            
            # For each token position, get the probability of the actual token
            # Logits at position i predict token at position i+1
            # So for token at position i (0-indexed), we use logits at position i-1
            token_log_probs = []
            for i, token_id in enumerate(token_ids):
                if i > 0 and i-1 < len(log_probs) and token_id < len(log_probs[i-1]):
                    # Use logits from previous position to predict current token
                    token_log_probs.append(log_probs[i-1][token_id])
                elif i == 0:
                    # For first token, we don't have previous context in the logits
                    # Skip first token or use uniform prior (we'll skip for now)
                    continue
            
            if len(token_log_probs) == 0:
                perplexities.append(float('inf'))
            else:
                avg_log_prob = np.mean(token_log_probs)
                perplexity = np.exp(-avg_log_prob)
                perplexities.append(float(perplexity))

        return perplexities[0] if is_single else perplexities


class BLEUReward(BaseReward):
    """Compute BLEU score."""

    def __init__(self, n_gram: int = 4):
        """Initialize BLEU reward.
        
        Args:
            n_gram: N-gram order for BLEU computation
        """
        super().__init__(f"bleu-{n_gram}")
        self.n_gram = n_gram
        if not BLEU_AVAILABLE:
            raise ImportError("BLEU requires nltk. Install with: pip install nltk && python -m nltk.downloader punkt")

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[float, List[float]]:
        """Compute BLEU score.
        
        Args:
            predictions: Predicted text(s)
            references: Reference text(s) or list of reference lists
            **kwargs: Additional arguments
            
        Returns:
            BLEU score(s)
        """
        is_single = isinstance(predictions, str)
        if is_single:
            predictions = [predictions]
        
        # Normalize references format
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                references = [[ref] for ref in references]
        
        smoothing = SmoothingFunction().method1
        scores = []
        
        for pred, refs in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens_list = [word_tokenize(ref.lower()) for ref in refs]
            
            score = sentence_bleu(
                ref_tokens_list,
                pred_tokens,
                smoothing_function=smoothing,
                weights=tuple([1.0 / self.n_gram] * self.n_gram)
            )
            scores.append(float(score))

        return scores[0] if is_single else scores


class ROUGEReward(BaseReward):
    """Compute ROUGE scores."""

    def __init__(self, rouge_types: List[str] = None):
        """Initialize ROUGE reward.
        
        Args:
            rouge_types: List of ROUGE types to compute (default: ['rouge1', 'rouge2', 'rougeL'])
        """
        super().__init__("rouge")
        if not ROUGE_AVAILABLE:
            raise ImportError("ROUGE requires rouge-score. Install with: pip install rouge-score")
        
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Compute ROUGE scores.
        
        Args:
            predictions: Predicted text(s)
            references: Reference text(s) or list of reference lists
            **kwargs: Additional arguments
            
        Returns:
            ROUGE scores as dict or list of dicts
        """
        is_single = isinstance(predictions, str)
        if is_single:
            predictions = [predictions]
        
        # Normalize references format
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                references = [[ref] for ref in references]
        
        all_scores = []
        
        for pred, refs in zip(predictions, references):
            # Use first reference (can be extended to use best match)
            ref = refs[0] if refs else ""
            scores = self.scorer.score(ref, pred)
            
            # Extract f-measure for each ROUGE type
            result = {}
            for rouge_type in self.rouge_types:
                result[rouge_type] = scores[rouge_type].fmeasure
            all_scores.append(result)

        return all_scores[0] if is_single else all_scores


class ExactMatchReward(BaseReward):
    """Compute exact match score."""

    def __init__(self, normalize: bool = True):
        """Initialize exact match reward.
        
        Args:
            normalize: Whether to normalize text (lowercase, strip whitespace)
        """
        super().__init__("exact_match")
        self.normalize = normalize

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.normalize:
            return text.lower().strip()
        return text.strip()

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[float, List[float]]:
        """Compute exact match score.
        
        Args:
            predictions: Predicted text(s)
            references: Reference text(s) or list of reference lists
            **kwargs: Additional arguments
            
        Returns:
            Exact match score(s) (1.0 for match, 0.0 otherwise)
        """
        is_single = isinstance(predictions, str)
        if is_single:
            predictions = [predictions]
        
        # Normalize references format
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                references = [[ref] for ref in references]
        
        scores = []
        
        for pred, refs in zip(predictions, references):
            pred_norm = self._normalize_text(pred)
            match = any(self._normalize_text(ref) == pred_norm for ref in refs)
            scores.append(1.0 if match else 0.0)

        return scores[0] if is_single else scores



class MathVerifyReward(BaseReward):
    """Compute math verification score using math_verify library.
    
    Extracts the final answer from the solution text and verifies
    if it's mathematically equivalent to the reference answer.
    """

    def __init__(self):
        """Initialize math verify reward."""
        super().__init__("math_verify")
        if not MATH_VERIFY_AVAILABLE:
            raise ImportError(
                "math_verify is not installed. Install it with: pip install math-verify"
            )

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from solution text.
        
        This method looks for common patterns like:
        - "The answer is X"
        - "Answer: X"
        - "X" (if text is just a number/expression)
        - Boxed answers: "\\boxed{X}"
        - Final line containing a number/expression
        
        Args:
            text: Solution text
            
        Returns:
            Extracted answer string or None if not found
        """
        import re
        
        # Remove extra whitespace
        text = text.strip()
        
        # Try to find boxed answer: \boxed{...} or \boxed ...
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, text)
        if match:
            return match.group(1).strip()
        
        # Try patterns like "The answer is X" or "Answer: X"
        answer_patterns = [
            r'(?:the\s+)?answer\s+is\s*:?\s*([^\n\.]+)',
            r'answer\s*:?\s*([^\n\.]+)',
            r'final\s+answer\s*:?\s*([^\n\.]+)',
            r'solution\s*:?\s*([^\n\.]+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Remove trailing punctuation
                answer = answer.rstrip('.,;!?')
                if answer:
                    return answer
        
        # Try to find the last number or mathematical expression
        # Look for patterns like numbers, fractions, expressions
        number_patterns = [
            r'(-?\d+\.?\d*)',  # Simple number
            r'(-?\d+/\d+)',  # Fraction
            r'(-?\d+\s*[+\-*/]\s*\d+)',  # Simple expression
        ]
        
        # Get the last few lines
        lines = text.split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like an answer (contains numbers)
            for pattern in number_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    # Return the last match
                    return matches[-1]
        
        # If nothing found, try to return the last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                return line
        
        return None

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
        **kwargs
    ) -> Union[float, List[float]]:
        """Compute math verification score.
        
        Args:
            predictions: Predicted solution text(s)
            references: Reference answer(s) or list of reference lists
            **kwargs: Additional arguments
            
        Returns:
            Math verification score(s) (1.0 if equivalent, 0.0 otherwise)
        """
        if not MATH_VERIFY_AVAILABLE:
            raise ImportError(
                "math_verify is not installed. Install it with: pip install math-verify"
            )
        
        is_single = isinstance(predictions, str)
        if is_single:
            predictions = [predictions]
        
        # Normalize references format
        if isinstance(references, str):
            references = [[references]]
        elif isinstance(references, list) and len(references) > 0:
            if isinstance(references[0], str):
                references = [[ref] for ref in references]
        
        scores = []
        
        for pred, refs in zip(predictions, references):
            # Extract answer from prediction
            pred_answer = self._extract_answer(pred)
            
            if pred_answer is None:
                # If we can't extract an answer, score is 0
                scores.append(0.0)
                continue
            
            # Try to verify against each reference answer
            verified = False
            for ref in refs:
                try:
                    # Use math_verify to check equivalence
                    result = verify(pred_answer, ref)
                    if result:
                        verified = True
                        break
                except Exception as e:
                    # If verification fails, continue to next reference
                    continue
            
            scores.append(1.0 if verified else 0.0)
        
        return scores[0] if is_single else scores
