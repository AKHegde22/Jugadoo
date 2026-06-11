"""
Structured JSONL logger for JugaadReasoning-1K evaluation results.

Provides atomic writes (write-to-temp, then rename/append), incremental
checkpointing for crash recovery, and typed deserialization back into
Pydantic models.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from jugaad_bench.models import CompletionResult, EvalResult

logger = logging.getLogger(__name__)


class ResultLogger:
    """
    Writes and reads structured JSONL files for evaluation results.

    Args:
        output_path: Path to the primary JSONL results file.
        checkpoint_dir: Directory for per-model completion checkpoints.
            Defaults to ``{output_path.parent}/checkpoints``.
    """

    def __init__(
        self,
        output_path: Path,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._checkpoint_dir = checkpoint_dir or (
            self._output_path.parent / "checkpoints"
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── EvalResult logging ──────────────────────────────────────────────

    def log_eval_result(self, result: EvalResult) -> None:
        """
        Atomically append an ``EvalResult`` to the results JSONL.

        The record is first written to a temporary file in the same
        directory, then its content is appended to the main file.
        This avoids partial writes on crash.

        Args:
            result: Fully populated evaluation result.
        """
        record = result.model_dump_json()
        self._atomic_append(self._output_path, record + "\n")
        logger.info(
            "Logged EvalResult for %s → %s",
            result.model_under_test,
            self._output_path,
        )

    def load_all_results(self) -> list[EvalResult]:
        """
        Read back all ``EvalResult`` records from the results JSONL.

        Returns:
            List of ``EvalResult`` instances in file order.
        """
        if not self._output_path.exists():
            return []

        results: list[EvalResult] = []
        with open(self._output_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(EvalResult.model_validate_json(line))
                except Exception as exc:
                    logger.warning(
                        "Skipping malformed record at line %d in %s: %s",
                        lineno,
                        self._output_path,
                        exc,
                    )
        logger.info(
            "Loaded %d EvalResult records from %s", len(results), self._output_path
        )
        return results

    # ── Checkpoint (per-completion) logging ──────────────────────────────

    def _checkpoint_path(self, model_name: str) -> Path:
        """Return the checkpoint file path for *model_name*."""
        safe = re.sub(r"[^\w\-]", "_", model_name)
        return self._checkpoint_dir / f"{safe}.jsonl"

    def log_completion(self, result: CompletionResult) -> None:
        """
        Append a single ``CompletionResult`` to the model-specific checkpoint.

        Args:
            result: A single completion record.
        """
        path = self._checkpoint_path(result.model_name)
        record = result.model_dump_json()
        self._atomic_append(path, record + "\n")

    def load_checkpoint(self, model_name: str) -> list[CompletionResult]:
        """
        Load all checkpointed completions for *model_name*.

        Args:
            model_name: Name of the model.

        Returns:
            List of ``CompletionResult`` records.
        """
        path = self._checkpoint_path(model_name)
        if not path.exists():
            return []

        results: list[CompletionResult] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(CompletionResult.model_validate_json(line))
                except Exception as exc:
                    logger.warning(
                        "Skipping malformed checkpoint at line %d in %s: %s",
                        lineno,
                        path,
                        exc,
                    )
        logger.info("Loaded %d completions from checkpoint %s", len(results), path)
        return results

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _atomic_append(path: Path, data: str) -> None:
        """
        Write *data* to a temp file, then append it to *path*.

        This ensures the main file never contains a partial JSON line, even
        if the process crashes mid-write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first
        fd, tmp_path_str = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".result_"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())

            # Append contents of temp file to main file
            with open(path, "a", encoding="utf-8") as main:
                main.write(tmp_path.read_text(encoding="utf-8"))
                main.flush()
                os.fsync(main.fileno())
        finally:
            tmp_path.unlink(missing_ok=True)
