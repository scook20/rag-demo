"""Per-request metrics: cost estimates from a pricing file, and an append-only JSONL log."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


def load_pricing(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def estimate_cost(
    pricing: dict[str, Any], model: str, tokens_in: int | None, tokens_out: int | None = None
) -> float | None:
    """USD estimate from per-1M-token prices. None when the model or token count is unknown."""
    entry = (pricing.get("models") or {}).get(model)
    if entry is None:
        return None
    cost = 0.0
    if tokens_in is not None:
        cost += tokens_in / 1_000_000 * float(entry.get("input", 0.0))
    if tokens_out is not None:
        cost += tokens_out / 1_000_000 * float(entry.get("output", 0.0))
    return cost


def strict_json_value(value: Any) -> Any:
    """Replace NaN and +/-Infinity (not valid JSON) with None, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: strict_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(v) for v in value]
    return value


def dumps_strict(record: Any, **kwargs: Any) -> str:
    """json.dumps that can only emit valid JSON: non-finite floats become null."""
    return json.dumps(strict_json_value(record), allow_nan=False, ensure_ascii=False, **kwargs)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps_strict(record, separators=(",", ":")) + "\n")
