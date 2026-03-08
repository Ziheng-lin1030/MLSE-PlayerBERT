#!/usr/bin/env python3
"""Reusable event feature helpers extracted from the notebooks."""

from __future__ import annotations

from typing import Any

import torch


UNK_TOKEN = "[UNK]"


def get_by_path(event: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Support both flattened dot-keys and nested dictionaries."""
    if dotted_key in event:
        return event[dotted_key]

    current: Any = event
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def normalize_feature_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return UNK_TOKEN
    return value


def build_feat_ids(ev: dict[str, Any], feature_vocab: dict[str, dict[Any, int]]) -> torch.Tensor:
    """Map a single flattened event into the exact feature-id vector used by the notebooks."""
    ids: list[int] = []
    for feat_name, vocab in feature_vocab.items():
        value = normalize_feature_value(get_by_path(ev, feat_name, UNK_TOKEN))
        idx = vocab.get(value, vocab.get(UNK_TOKEN, 0))
        if idx >= len(vocab):
            idx = vocab.get(UNK_TOKEN, 0)
        ids.append(idx)
    return torch.tensor(ids, dtype=torch.long)


def get_event_type(ev: dict[str, Any]) -> str | None:
    value = get_by_path(ev, "type.name")
    if value is None:
        return None
    return str(value)


def get_player_id(ev: dict[str, Any]) -> str | None:
    value = get_by_path(ev, "player.id")
    if value is None:
        return None
    return str(value)


def get_player_name(ev: dict[str, Any]) -> str | None:
    value = get_by_path(ev, "player.name")
    if value is None:
        return None
    return str(value)


def get_actor_loc(ev: dict[str, Any]) -> tuple[float, float]:
    location = get_by_path(ev, "location")
    if isinstance(location, list) and len(location) >= 2:
        try:
            return float(location[0]), float(location[1])
        except Exception:
            return 0.0, 0.0
    return 0.0, 0.0


def get_freeze_frame(ev: dict[str, Any]) -> list[Any]:
    freeze_frame = get_by_path(ev, "freeze_frame", [])
    if isinstance(freeze_frame, list):
        return freeze_frame
    return []
