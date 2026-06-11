"""
Tests for the keyword guard filter.

Validates stemming, whole-word matching, case insensitivity,
and correct AUTO_FAIL / CLEAN classification.
"""

import pytest

from jugaad_bench.models import FilterResult


class TestKeywordGuard:
    """Tests for the KeywordGuard class."""

    @pytest.fixture
    def guard(self):
        from jugaad_bench.eval.keyword_guard import KeywordGuard
        return KeywordGuard()

    # --- Basic Detection ---

    def test_clean_response(self, guard):
        """A response with no forbidden keywords should be CLEAN."""
        result = guard.check(
            response="Fill the IV bottle with water and hang it upside down on the bamboo stake.",
            forbidden_keywords=["buy", "online", "motor"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.CLEAN
        assert result.triggered_keywords == []

    def test_forbidden_keyword_detected(self, guard):
        """A response containing a forbidden keyword should AUTO_FAIL."""
        result = guard.check(
            response="The farmer should buy a drip irrigation kit from the local store.",
            forbidden_keywords=["buy", "online", "motor"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.AUTO_FAIL
        assert "buy" in result.triggered_keywords

    def test_multiple_forbidden_keywords(self, guard):
        """Multiple forbidden keywords should all be detected."""
        result = guard.check(
            response="Go online and buy a motor pump for irrigation.",
            forbidden_keywords=["buy", "online", "motor"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.AUTO_FAIL
        assert len(result.triggered_keywords) >= 2

    # --- Case Insensitivity ---

    def test_case_insensitive(self, guard):
        """Matching should be case-insensitive."""
        result = guard.check(
            response="The farmer should BUY a kit ONLINE.",
            forbidden_keywords=["buy", "online"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.AUTO_FAIL

    # --- Stemming ---

    def test_stemming_buying(self, guard):
        """'buying' should trigger 'buy' via stemming."""
        result = guard.check(
            response="Consider buying materials from the market.",
            forbidden_keywords=["buy"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.AUTO_FAIL

    def test_stemming_worked(self, guard):
        """'worked' should trigger 'work' via stemming."""
        result = guard.check(
            response="She worked on a kit from the store.",
            forbidden_keywords=["work"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.AUTO_FAIL

    def test_stemming_motorized(self, guard):
        """'electrical' should trigger 'electric' via stemming."""
        result = guard.check(
            response="Use an electrical pump for water extraction.",
            forbidden_keywords=["electric"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.AUTO_FAIL

    # --- Edge Cases ---

    def test_empty_response(self, guard):
        """An empty response should be CLEAN (no keywords to detect)."""
        result = guard.check(
            response="",
            forbidden_keywords=["buy", "online"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.CLEAN

    def test_empty_forbidden_list(self, guard):
        """No forbidden keywords means always CLEAN."""
        result = guard.check(
            response="Buy everything online with a motor.",
            forbidden_keywords=[],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.CLEAN

    def test_keyword_as_substring_in_different_word(self, guard):
        """'motor' should NOT trigger on 'motorcycle' since we do word boundary matching."""
        result = guard.check(
            response="Use a motorcycle to transport water.",
            forbidden_keywords=["motor"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.CLEAN

    def test_punctuation_handling(self, guard):
        """Keywords should be detected even adjacent to punctuation."""
        result = guard.check(
            response="Don't buy, order, or use any motor-driven device.",
            forbidden_keywords=["buy", "motor"],
            problem_id="test-001",
            model_name="test-model",
        )
        assert result.filter_result == FilterResult.AUTO_FAIL
        assert "buy" in result.triggered_keywords or "motor" in result.triggered_keywords
