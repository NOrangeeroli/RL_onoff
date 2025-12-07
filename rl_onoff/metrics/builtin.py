"""Built-in metrics for evaluating model outputs."""

from typing import Union, List, Dict, Optional
import numpy as np

from rl_onoff.metrics.base import BaseMetric

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


class PerplexityMetric(BaseMetric):
    """Compute perplexity of generated text."""

    def __init__(self, backend=None):
        """Initialize perplexity metric.
        
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
            
            # Get logits for the sequence
            logits = self.backend.get_logits(pred, max_new_tokens=len(token_ids))
            
            # Compute log probabilities
            probs = self.backend.get_probabilities(pred, max_new_tokens=len(token_ids))
            
            # Compute perplexity: exp(-1/N * sum(log P(token_i)))
            log_probs = np.log(probs + 1e-10)  # Add small epsilon for numerical stability
            
            # For each position, get the probability of the actual token
            token_log_probs = []
            for i, token_id in enumerate(token_ids):
                if i < len(log_probs):
                    token_log_probs.append(log_probs[i][token_id])
            
            if len(token_log_probs) == 0:
                perplexities.append(float('inf'))
            else:
                avg_log_prob = np.mean(token_log_probs)
                perplexity = np.exp(-avg_log_prob)
                perplexities.append(float(perplexity))

        return perplexities[0] if is_single else perplexities


class BLEUMetric(BaseMetric):
    """Compute BLEU score."""

    def __init__(self, n_gram: int = 4):
        """Initialize BLEU metric.
        
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


class ROUGEMetric(BaseMetric):
    """Compute ROUGE scores."""

    def __init__(self, rouge_types: List[str] = None):
        """Initialize ROUGE metric.
        
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


class ExactMatchMetric(BaseMetric):
    """Compute exact match score."""

    def __init__(self, normalize: bool = True):
        """Initialize exact match metric.
        
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

