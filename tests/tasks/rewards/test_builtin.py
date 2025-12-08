"""Tests for built-in rewards."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from rl_onoff.tasks.rewards.builtin import (
    ExactMatchReward,
    MathVerifyReward,
)


class TestExactMatchReward:
    """Tests for ExactMatchReward."""
    
    def test_init_default(self):
        """Test initialization with default normalize=True."""
        reward = ExactMatchReward()
        assert reward.name == "exact_match"
    
    def test_init_normalize_false(self):
        """Test initialization with normalize=False."""
        reward = ExactMatchReward(normalize=False)
        assert reward.name == "exact_match"
    
    def test_compute_exact_match_single(self):
        """Test compute with exact match (single)."""
        reward = ExactMatchReward()
        result = reward.compute("hello", "hello")
        assert result == 1.0
    
    def test_compute_no_match_single(self):
        """Test compute with no match (single)."""
        reward = ExactMatchReward()
        result = reward.compute("hello", "world")
        assert result == 0.0
    
    def test_compute_exact_match_list(self):
        """Test compute with exact matches (list)."""
        reward = ExactMatchReward()
        result = reward.compute(["hello", "world"], ["hello", "world"])
        assert result == [1.0, 1.0]
    
    def test_compute_partial_match_list(self):
        """Test compute with partial matches (list)."""
        reward = ExactMatchReward()
        result = reward.compute(["hello", "world"], ["hello", "foo"])
        assert result == [1.0, 0.0]
    
    def test_compute_normalize_true(self):
        """Test compute with normalization enabled."""
        reward = ExactMatchReward(normalize=True)
        result = reward.compute("  Hello  ", "hello")
        assert result == 1.0
    
    def test_compute_normalize_false(self):
        """Test compute with normalization disabled."""
        reward = ExactMatchReward(normalize=False)
        result = reward.compute("  Hello  ", "hello")
        assert result == 0.0  # No match due to whitespace
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="BLEU not available")
    def test_bleu_reward_available(self):
        """Test BLEUReward if available."""
        try:
            from rl_onoff.tasks.rewards.builtin import BLEUReward
            reward = BLEUReward()
            result = reward.compute("hello world", "hello world")
            assert isinstance(result, (float, int))
        except ImportError:
            pytest.skip("BLEU dependencies not available")
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="ROUGE not available")
    def test_rouge_reward_available(self):
        """Test ROUGEReward if available."""
        try:
            from rl_onoff.tasks.rewards.builtin import ROUGEReward
            reward = ROUGEReward()
            result = reward.compute("hello world", "hello world")
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("ROUGE dependencies not available")


class TestMathVerifyReward:
    """Tests for MathVerifyReward."""
    
    def test_init(self):
        """Test initialization."""
        reward = MathVerifyReward()
        assert reward.name == "math_verify"
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="math_verify not available")
    def test_compute_with_math_verify_available(self):
        """Test compute when math_verify is available."""
        try:
            from math_verify import verify
            reward = MathVerifyReward()
            # Test with simple case
            result = reward.compute("2+2", "4")
            assert isinstance(result, (float, int))
            assert 0.0 <= result <= 1.0
        except ImportError:
            pytest.skip("math_verify not available")
    
    def test_compute_without_math_verify(self):
        """Test compute when math_verify is not available."""
        with patch('rl_onoff.tasks.rewards.builtin.MATH_VERIFY_AVAILABLE', False):
            reward = MathVerifyReward()
            with pytest.raises(ImportError):
                reward.compute("2+2", "4")
    
    def test_extract_answer_from_text(self):
        """Test _extract_answer method."""
        try:
            from rl_onoff.tasks.rewards.builtin import MathVerifyReward
            reward = MathVerifyReward()
            
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

