"""Tests for built-in rewards."""

import pytest
from unittest.mock import MagicMock, patch
import importlib
import numpy as np
from rl_onoff.tasks.rewards import builtin


class TestExactMatchReward:
    """Tests for ExactMatchReward."""
    
    def test_init_default(self):
        """Test initialization with default normalize=True."""
        reward = builtin.ExactMatchReward()
        assert reward.name == "exact_match"
    
    def test_init_normalize_false(self):
        """Test initialization with normalize=False."""
        reward = builtin.ExactMatchReward(normalize=False)
        assert reward.name == "exact_match"
    
    def test_compute_exact_match_single(self):
        """Test compute with exact match (single)."""
        reward = builtin.ExactMatchReward()
        result = reward.compute("hello", "hello")
        assert result == 1.0
    
    def test_compute_no_match_single(self):
        """Test compute with no match (single)."""
        reward = builtin.ExactMatchReward()
        result = reward.compute("hello", "world")
        assert result == 0.0
    
    def test_compute_exact_match_list(self):
        """Test compute with exact matches (list)."""
        reward = builtin.ExactMatchReward()
        result = reward.compute(["hello", "world"], ["hello", "world"])
        assert result == [1.0, 1.0]
    
    def test_compute_partial_match_list(self):
        """Test compute with partial matches (list)."""
        reward = builtin.ExactMatchReward()
        result = reward.compute(["hello", "world"], ["hello", "foo"])
        assert result == [1.0, 0.0]
    
    def test_compute_normalize_true(self):
        """Test compute with normalization enabled."""
        reward = builtin.ExactMatchReward(normalize=True)
        result = reward.compute("  Hello  ", "hello")
        assert result == 1.0
    
    def test_compute_normalize_false(self):
        """Test compute with normalization disabled."""
        reward = builtin.ExactMatchReward(normalize=False)
        result = reward.compute("  Hello  ", "hello")
        assert result == 0.0  # No match due to whitespace
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="BLEU not available")
    def test_bleu_reward_available(self):
        """Test BLEUReward if available."""
        try:
            reward = builtin.BLEUReward()
            result = reward.compute("hello world", "hello world")
            assert isinstance(result, (float, int))
        except ImportError:
            pytest.skip("BLEU dependencies not available")
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="ROUGE not available")
    def test_rouge_reward_available(self):
        """Test ROUGEReward if available."""
        try:
            reward = builtin.ROUGEReward()
            result = reward.compute("hello world", "hello world")
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("ROUGE dependencies not available")


class TestMathVerifyReward:
    """Tests for MathVerifyReward."""
    
    def test_init(self):
        """Test initialization."""
        try:
            reward = builtin.MathVerifyReward()
            assert reward.name == "math_verify"
        except ImportError:
            pytest.skip("math_verify not available")
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="math_verify not available")
    def test_compute_with_math_verify_available(self):
        """Test compute when math_verify is available."""
        try:
            reward = builtin.MathVerifyReward()
            # Test with simple case
            result = reward.compute("2+2", "4")
            assert isinstance(result, (float, int))
            assert 0.0 <= result <= 1.0
        except ImportError:
            pytest.skip("math_verify not available")
    
    def test_compute_without_math_verify(self):
        """Test compute when math_verify is not available."""
        # Skip this test if math_verify is actually not available (can't test the error case)
        if not builtin.MATH_VERIFY_AVAILABLE:
            pytest.skip("math_verify is not available, cannot test error handling")
        
        # Use patch to temporarily set MATH_VERIFY_AVAILABLE to False
        with patch.object(builtin, 'MATH_VERIFY_AVAILABLE', False):
            # The class checks MATH_VERIFY_AVAILABLE in __init__, so this should work
            with pytest.raises(ImportError, match="math_verify is not installed"):
                reward = builtin.MathVerifyReward()
    
    def test_extract_answer_from_text(self):
        """Test _extract_answer method."""
        try:
            reward = builtin.MathVerifyReward()
            
            # Test with boxed answer
            answer = reward._extract_answer("The solution is \\boxed{42}")
            assert answer == "42"
            
            # Test with "answer is" pattern
            answer = reward._extract_answer("The answer is 42")
            assert answer == "42"
            
            # Test with "Answer:" pattern
            answer = reward._extract_answer("Answer: 42")
            assert answer == "42"
        except ImportError:
            pytest.skip("MathVerifyReward not available")

