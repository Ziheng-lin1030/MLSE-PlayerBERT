#!/usr/bin/env python3
"""Preprocess StatsBomb 360 frames by joining to events via event_uuid.

Creates one row per 360 frame with event attributes + visible_area + freeze_frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import copy
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []


def iter_json_files(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.glob("*.json") if p.is_file())


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_lineup_position_map(lineups_data: list[dict]) -> dict[int, dict]:
    """Build a player_id -> position dict from lineups data.

    Uses the first listed position for each player.
    """
    positions: dict[int, dict] = {}
    for team in lineups_data:
        for player in team.get("lineup", []):
            pid = player.get("player_id")
            pos_list = player.get("positions") or []
            if pid is None or not pos_list:
                continue
            pos = pos_list[0]
            pos_id = pos.get("position_id")
            pos_name = pos.get("position")
            if pos_id is None and pos_name is None:
                continue
            positions[int(pid)] = {
                "id": pos_id,
                "name": pos_name,
            }
    return positions


def compute_zone(location: list[float] | None, x_bins: int = 6, y_bins: int = 3) -> dict | None:
    """Compute a coarse pitch zone for a location.

    Returns a dict with x_bin, y_bin, and label.
    """
    if location is None or len(location) < 2:
        return None
    x, y = float(location[0]), float(location[1])
    # Clamp to pitch bounds
    x = max(0.0, min(120.0, x))
    y = max(0.0, min(80.0, y))
    x_bin = min(x_bins - 1, int(x / (120.0 / x_bins)))
    y_bin = min(y_bins - 1, int(y / (80.0 / y_bins)))
    return {
        "x_bin": x_bin,
        "y_bin": y_bin,
        "label": f"{x_bin}_{y_bin}",
    }


def bucketize_location(location: list[float] | None, x_bins: int = 6, y_bins: int = 3) -> dict | None:
    """Bucketize a pitch location into a coarse zone."""
    return compute_zone(location, x_bins=x_bins, y_bins=y_bins)


def bucketize_length(length: float | None) -> str | None:
    """Bucketize pass length (yards)."""
    if length is None:
        return None
    if length < 10.0:
        return "0-10"
    if length < 25.0:
        return "10-25"
    if length < 40.0:
        return "25-40"
    return "40+"


def bucketize_angle(angle: float | None) -> str | None:
    """Bucketize pass angle (radians)."""
    if angle is None:
        return None
    if angle < -0.35:
        return "left"
    if angle > 0.35:
        return "right"
    return "center"


def bucketize_xg(xg: float | None) -> str | None:
    """Bucketize xG into coarse bins."""
    if xg is None:
        return None
    if xg < 0.05:
        return "0-0.05"
    if xg < 0.15:
        return "0.05-0.15"
    if xg < 0.30:
        return "0.15-0.30"
    return "0.30+"


def normalize_booleans(details: dict, bool_keys: list[str]) -> None:
    """Ensure boolean keys exist in a dict; missing -> False."""
    for key in bool_keys:
        if key not in details or details.get(key) is None:
            details[key] = False


def normalize_categoricals(details: dict, cat_keys: list[str]) -> None:
    """Ensure categorical keys exist in a dict; missing -> Unknown."""
    for key in cat_keys:
        if key not in details or details.get(key) is None:
            details[key] = {"id": None, "name": "Unknown"}


def build_rows(
    events_dir: Path,
    frames_dir: Path,
    lineups_dir: Path,
    max_matches: int | None = None,
) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """Join 360 frames to events match-by-match. Returns rows and errors.
    
    Args:
        events_dir: Directory with event JSON files
        frames_dir: Directory with 360 frame JSON files  
        lineups_dir: Directory with lineups JSON files
        max_matches: Maximum number of matches to process (None = all)
    """
    rows: List[dict] = []
    errors: List[Tuple[str, str]] = []

    # Get list of 360 frame files
    frame_files = list(iter_json_files(frames_dir))
    if max_matches is not None:
        frame_files = frame_files[:max_matches]
    
    for frame_fp in tqdm(frame_files, desc="Processing matches"):
        match_id = frame_fp.stem
        event_fp = events_dir / f"{match_id}.json"
        
        # Load lineups for this match (optional)
        lineup_positions: dict[int, dict] = {}
        lineup_fp = lineups_dir / f"{match_id}.json"
        if lineup_fp.exists():
            try:
                lineups_data = load_json(lineup_fp)
                lineup_positions = build_lineup_position_map(lineups_data)
            except Exception as exc:  # noqa: BLE001
                errors.append((match_id, f"Error loading lineups: {exc}"))

        # Load events for this match
        if not event_fp.exists():
            errors.append((match_id, f"Event file not found: {event_fp}"))
            continue
            
        try:
            events_data = load_json(event_fp)
        except Exception as exc:  # noqa: BLE001
            errors.append((match_id, f"Error loading events: {exc}"))
            continue
        
        # Build event index for this match only
        event_index = {}
        for event in events_data:
            event_id = event.get("id") or event.get("event_uuid")
            if event_id:
                event_index[str(event_id)] = event
        
        # Load 360 frames for this match
        try:
            frames_data = load_json(frame_fp)
        except Exception as exc:  # noqa: BLE001
            errors.append((match_id, f"Error loading frames: {exc}"))
            continue
        
        # Join frames to events
        for frame in frames_data:
            event_uuid = frame.get("event_uuid")
            event = event_index.get(str(event_uuid), {})
            row = copy.deepcopy(event)  # event attributes
            # Sanitize pass/shot fields to reduce unique IDs
            p = row.get("pass")
            if isinstance(p, dict):
                # Remove unique assisted_shot_id
                p.pop("assisted_shot_id", None)
                # Replace recipient with recipient_position (from lineups)
                recipient = p.pop("recipient", None)
                if isinstance(recipient, dict):
                    rid = recipient.get("id")
                    if rid is not None and int(rid) in lineup_positions:
                        p["recipient_position"] = lineup_positions[int(rid)]
                # If no lineup info, omit recipient_position entirely
                # Normalize booleans and categoricals
                normalize_booleans(
                    p,
                    [
                        "backheel",
                        "deflected",
                        "miscommunication",
                        "cross",
                        "cut_back",
                        "switch",
                        "shot_assist",
                        "goal_assist",
                    ],
                )
                normalize_categoricals(p, ["body_part", "type", "outcome", "height", "technique"])

                # Bucketize pass length/angle and end_location
                length = p.get("length")
                if length is not None:
                    try:
                        p["length_bucket"] = bucketize_length(float(length))
                    except Exception:
                        p["length_bucket"] = None
                else:
                    p["length_bucket"] = None

                angle = p.get("angle")
                if angle is not None:
                    try:
                        p["angle_bucket"] = bucketize_angle(float(angle))
                    except Exception:
                        p["angle_bucket"] = None
                else:
                    p["angle_bucket"] = None

                p["end_location_bucket"] = bucketize_location(p.get("end_location"))

            s = row.get("shot")
            if isinstance(s, dict):
                key_pass_id = s.pop("key_pass_id", None)
                if key_pass_id is not None:
                    key_pass_event = event_index.get(str(key_pass_id)) or event_index.get(key_pass_id)
                    if isinstance(key_pass_event, dict):
                        zone = compute_zone(key_pass_event.get("location"))
                        if zone is not None:
                            s["key_pass_zone"] = zone
                # Normalize booleans and categoricals
                normalize_booleans(
                    s,
                    [
                        "aerial_won",
                        "follows_dribble",
                        "first_time",
                        "open_goal",
                        "deflected",
                    ],
                )
                normalize_categoricals(s, ["body_part", "type", "outcome", "technique"])

                # Bucketize xG and end_location
                xg = s.get("statsbomb_xg")
                if xg is not None:
                    try:
                        s["xg_bucket"] = bucketize_xg(float(xg))
                    except Exception:
                        s["xg_bucket"] = None
                else:
                    s["xg_bucket"] = None

                s["end_location_bucket"] = bucketize_location(s.get("end_location"))

            # Bucketize top-level location
            row["location_bucket"] = bucketize_location(row.get("location"))

            row["event_uuid"] = event_uuid
            row["match_id"] = match_id
            row["visible_area"] = frame.get("visible_area")
            row["freeze_frame"] = frame.get("freeze_frame")
            rows.append(row)

    return rows, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join 360 frames to events.")
    parser.add_argument(
        "--events_dir",
        type=Path,
        default=Path("open-data/data/events"),
        help="Directory with StatsBomb events JSON files",
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=Path("open-data/data/three-sixty"),
        help="Directory with StatsBomb 360 frames JSON files",
    )
    parser.add_argument(
        "--lineups_dir",
        type=Path,
        default=Path("open-data/data/lineups"),
        help="Directory with StatsBomb lineups JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("open-data/data/processed/events360_v2.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--max_matches",
        type=int,
        default=None,
        help="Maximum number of matches to process (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_dir = args.events_dir
    frames_dir = args.frames_dir
    lineups_dir = args.lineups_dir
    output_path = args.output

    if not events_dir.exists():
        raise FileNotFoundError(f"Events directory not found: {events_dir}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    rows, errors = build_rows(events_dir, frames_dir, lineups_dir, max_matches=args.max_matches)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear any existing file to avoid appending
    if output_path.exists():
        output_path.unlink()
    df = pd.DataFrame(rows)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)

    print(f"Matches processed: {args.max_matches or 'all'}")
    print(f"Frames joined: {len(rows)}")
    print(f"Errors: {len(errors)}")
    if errors:
        print("First error:", errors[0])
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
