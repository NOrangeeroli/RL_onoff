"""Math verification reward using math_verify library."""

from typing import Union, List

from rl_onoff.tasks.rewards.base import BaseReward

try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    parse = None
    verify = None


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

    def compute(
        self,
        predictions: Union[str, List[str]],
        references: Union[str, List[str], List[List[str]]],
    ) -> Union[float, List[float]]:
        """Compute math verification score.
        
        Args:
            predictions: Predicted answer(s) (assumed to be already extracted)
            references: Reference answer(s) or list of reference lists
            
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
            # Use prediction directly as answer (assumed to be already extracted)
            pred_answer = pred.strip() if pred else None
            
            if not pred_answer:
                # If answer is empty, score is 0
                scores.append(0.0)
                continue
            
            # Try to verify against each reference answer
            verified = False
            for ref in refs:
                try:
                    # Parse both gold (reference) and answer (prediction) first
                    gold_parsed = parse(ref.strip())
                    answer_parsed = parse(pred_answer)
                    
                    # Verify: order is important! verify(gold, answer)
                    result = verify(gold_parsed, answer_parsed)
                    if result:
                        verified = True
                        break
                except Exception:
                    # If parsing or verification fails, continue to next reference
                    continue
            
            scores.append(1.0 if verified else 0.0)
        
        return scores[0] if is_single else scores


if __name__ == "__main__":
    # Simple usage example for MathVerifyReward
    print("MathVerifyReward Usage Example")
    print("=" * 50)
    
    try:
        # Create a MathVerifyReward instance
        reward = MathVerifyReward()
        
        # Example 1: Single prediction and reference (exact match)
        print("\nExample 1: Single answer verification")
        prediction = "42"
        reference = "42"
        score = reward.compute(prediction, reference)
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")
        print(f"Score: {score}")
        
        # Example 2: Equivalent set expressions
        print("\nExample 2: Equivalent set expressions")
        prediction = "${1,2,3,4}$"
        reference = "${1,3} \\cup {2,4}$"
        score = reward.compute(prediction, reference)
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")
        print(f"Score: {score}")
        
        # Example 3: Equivalent arithmetic expressions
        print("\nExample 3: Equivalent arithmetic expressions")
        prediction = "4"
        reference = "2 + 2"
        score = reward.compute(prediction, reference)
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")
        print(f"Score: {score}")
        
        # Example 4: Multiple predictions and references
        print("\nExample 4: Multiple answers")
        predictions = ["10", "15", "20"]
        references = ["5 * 2", "3 * 5", "4 * 5"]
        scores = reward.compute(predictions, references)
        print(f"Predictions: {predictions}")
        print(f"References: {references}")
        print(f"Scores: {scores}")
        
        # Example 5: Incorrect answer
        print("\nExample 5: Incorrect answer")
        prediction = "5"
        reference = "10"
        score = reward.compute(prediction, reference)
        print(f"Prediction: {prediction}")
        print(f"Reference: {reference}")
        print(f"Score: {score}")
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("To use MathVerifyReward, install math-verify:")
        print("  pip install math-verify")

