"""
Forbidden keyword filter – first gate in the JugaadReasoning-1K verification protocol.

Tokenizes responses and forbidden keywords, applies simple suffix-stripping stemming,
and checks for whole-word matches (word-boundary aware). Returns KeywordGuardResult.
"""

from __future__ import annotations

import logging
import re

from jugaad_bench.models import FilterResult, KeywordGuardResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Simple suffix-stripping stemmer
# ─────────────────────────────────────────────────────────────────────────────

# Ordered from longest suffix to shortest to avoid premature short-suffix removal.
_SUFFIX_RULES: list[tuple[str, int]] = [
    ("isation", 4),   # keep at least 4 chars after stripping
    ("ization", 4),
    ("ation", 4),
    ("tion", 4),
    ("sion", 4),
    ("ment", 4),
    ("ness", 4),
    ("able", 4),
    ("ible", 4),
    ("ance", 4),
    ("ence", 4),
    ("ment", 4),
    ("ings", 3),
    ("ally", 4),
    ("ful", 3),
    ("ous", 3),
    ("ive", 3),
    ("ing", 3),
    ("ise", 3),
    ("ize", 3),
    ("ate", 3),
    ("ion", 3),
    ("ly", 3),
    ("ed", 3),
    ("er", 3),
    ("es", 3),
    ("al", 3),
    ("s", 3),
]

# Pattern to tokenize on word boundaries: splits on non-alphanumeric characters
_TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+")


class KeywordGuard:
    """
    Checks model responses against a list of forbidden keywords.

    Uses suffix-stripping stemming and word-boundary tokenization to handle
    morphological variants while avoiding false positives from substring matches.
    """

    def __init__(self) -> None:
        """Initialize the guard (stateless; suffix rules are module-level constants)."""
        pass

    @staticmethod
    def stem(word: str) -> str:
        """
        Apply simple suffix stripping to *word*.

        Strips the longest matching suffix provided the remaining stem length
        meets the minimum threshold.

        Args:
            word: A single lowercase token.

        Returns:
            The stemmed form.
        """
        w = word.lower()
        for suffix, min_stem_len in _SUFFIX_RULES:
            if w.endswith(suffix) and len(w) - len(suffix) >= min_stem_len:
                return w[: -len(suffix)]
        return w

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Extract alphanumeric tokens from *text*."""
        return _TOKENIZE_RE.findall(text.lower())

    def check(
        self,
        response: str,
        forbidden_keywords: list[str],
        *,
        problem_id: str = "",
        model_name: str = "",
    ) -> KeywordGuardResult:
        """
        Check *response* for forbidden keywords (case-insensitive, stemmed).

        Multi-word forbidden entries (e.g., "electric motor") are handled by
        checking whether **all** stemmed sub-tokens of the entry appear in
        the response token bag.

        Args:
            response: The raw model output.
            forbidden_keywords: List of forbidden terms from the rubric.
            problem_id: For populating the result model.
            model_name: For populating the result model.

        Returns:
            A ``KeywordGuardResult`` indicating CLEAN or AUTO_FAIL.
        """
        response_tokens = self._tokenize(response)
        stemmed_response_tokens = {self.stem(t) for t in response_tokens}

        triggered: list[str] = []

        for keyword in forbidden_keywords:
            keyword_parts = self._tokenize(keyword)
            if not keyword_parts:
                continue

            stemmed_parts = [self.stem(p) for p in keyword_parts]

            # For a multi-word forbidden phrase, every stemmed sub-token must
            # be present in the response.  For a single-word term, that
            # degenerates to a simple membership check.
            if all(sp in stemmed_response_tokens for sp in stemmed_parts):
                triggered.append(keyword)
                logger.debug(
                    "Keyword guard triggered: '%s' (stems %s) found in %s / %s",
                    keyword,
                    stemmed_parts,
                    problem_id,
                    model_name,
                )

        filter_result = FilterResult.AUTO_FAIL if triggered else FilterResult.CLEAN

        if triggered:
            logger.info(
                "KeywordGuard AUTO_FAIL for %s/%s — triggered: %s",
                model_name,
                problem_id,
                triggered,
            )

        return KeywordGuardResult(
            problem_id=problem_id,
            model_name=model_name,
            filter_result=filter_result,
            triggered_keywords=triggered,
        )
