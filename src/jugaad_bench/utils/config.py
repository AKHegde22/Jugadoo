"""
Configuration loader for the JugaadReasoning-1K pipeline.

Loads YAML config and environment variables, returning typed PipelineConfig.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from jugaad_bench.models import PipelineConfig


# Default config path relative to project root
_DEFAULT_CONFIG = "configs/pipeline_config.yaml"


def find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml found in parent directories)."
    )


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """
    Load and validate the pipeline configuration.

    Args:
        config_path: Path to YAML config file. If None, uses default location.

    Returns:
        Validated PipelineConfig instance.
    """
    root = find_project_root()

    # Load .env file if present
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Resolve config path
    if config_path is None:
        config_path = root / _DEFAULT_CONFIG
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PipelineConfig.model_validate(raw)


def resolve_data_path(config: PipelineConfig, key: str) -> Path:
    """
    Resolve a data path from config relative to project root.

    Args:
        config: Pipeline configuration.
        key: Key in config.data.paths (e.g., 'seeds', 'mutations', 'benchmark').

    Returns:
        Absolute path to the data directory (created if it doesn't exist).
    """
    root = find_project_root()
    rel = config.data.paths.get(key, f"data/{key}")
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_api_key(provider: str, env_var: str | None = None) -> str:
    """
    Retrieve an API key for a given provider.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google').
        env_var: Override environment variable name.

    Returns:
        API key string.

    Raises:
        ValueError: If no API key is found.
    """
    default_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "youtube": "YOUTUBE_API_KEY",
        "krutrim": "KRUTRIM_API_KEY",
    }

    var_name = env_var or default_vars.get(provider.lower())
    if var_name is None:
        raise ValueError(
            f"Unknown provider '{provider}' and no env_var specified. "
            f"Known providers: {list(default_vars.keys())}"
        )

    key = os.environ.get(var_name)
    if not key:
        raise ValueError(
            f"API key not found. Set the {var_name} environment variable "
            f"(or add it to your .env file)."
        )

    return key
