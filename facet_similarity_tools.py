#!/usr/bin/env python3
"""Build and query Player x EventType similarity profiles from EventEncoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from playerbert_event_features import (
    build_feat_ids,
    get_actor_loc,
    get_event_type,
    get_freeze_frame,
    get_player_id,
    get_player_name,
)
from playerbert_models import load_event_encoder_checkpoint


DEFAULT_DATA_PATH = Path("open-data/data/processed/events360_v4.jsonl")
DEFAULT_EVENT_ENCODER_CKPT = Path("models/event_encoder_mam.pt")
DEFAULT_PROFILE_CACHE = Path("models/player_eventtype_profiles.pt")
DEFAULT_PLAYER_EMBEDDINGS = Path("models/player_embeddings.pt")
DEFAULT_PLAYER_EMBEDDING_NAMES = Path("models/player_embeddings_names.json")


def iter_json_objects(path: Path):
    """Read JSONL-like files robustly, including concatenated objects per line."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx = 0
            while idx < len(line):
                obj, end = decoder.raw_decode(line, idx)
                yield obj
                idx = end
                while idx < len(line) and line[idx].isspace():
                    idx += 1


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        requested = requested.strip().lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("Requested CUDA, but CUDA is unavailable in this environment. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixed_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except Exception:
        return 1, value


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def l2_normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def l2_normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-8)


def cosine_scores(matrix: torch.Tensor, query_vector: torch.Tensor) -> torch.Tensor:
    return l2_normalize_rows(matrix) @ l2_normalize_vector(query_vector)


def flush_embedding_batch(
    *,
    model,
    device: torch.device,
    batch_feat_ids: list[torch.Tensor],
    batch_freeze_frames: list[Any],
    batch_actor_locs: list[tuple[float, float]],
    batch_keys: list[tuple[str, str]],
    sum_vecs: dict[tuple[str, str], torch.Tensor],
    counts: dict[tuple[str, str], int],
) -> int:
    if not batch_feat_ids:
        return 0

    feat_tensor = torch.stack(batch_feat_ids, dim=0).to(device)
    with torch.no_grad():
        event_embeddings, _ = model(feat_tensor, batch_freeze_frames, batch_actor_locs, device)

    processed = 0
    for key, embedding in zip(batch_keys, event_embeddings.detach().cpu(), strict=False):
        embedding = embedding.float()
        if key in sum_vecs:
            sum_vecs[key].add_(embedding)
        else:
            sum_vecs[key] = embedding.clone()
        counts[key] = counts.get(key, 0) + 1
        processed += 1

    batch_feat_ids.clear()
    batch_freeze_frames.clear()
    batch_actor_locs.clear()
    batch_keys.clear()
    return processed


def save_profile_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_profile_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_player_eventtype_profiles(
    *,
    data_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int,
    min_count_to_store: int,
    device_name: str | None,
    unknown_event_type: str | None,
    log_every: int,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, feature_vocab, _ = load_event_encoder_checkpoint(checkpoint_path, device)

    sum_vecs: dict[tuple[str, str], torch.Tensor] = {}
    counts: dict[tuple[str, str], int] = {}
    player_names: dict[str, str] = {}
    player_ids_seen: set[str] = set()
    event_types_seen: set[str] = set()

    total_rows = 0
    encoded_rows = 0
    skipped_missing_player = 0
    skipped_missing_type = 0

    batch_feat_ids: list[torch.Tensor] = []
    batch_freeze_frames: list[Any] = []
    batch_actor_locs: list[tuple[float, float]] = []
    batch_keys: list[tuple[str, str]] = []

    for event in iter_json_objects(data_path):
        total_rows += 1

        player_id = get_player_id(event)
        if player_id is None:
            skipped_missing_player += 1
            continue

        event_type = get_event_type(event)
        if event_type is None:
            if unknown_event_type is None:
                skipped_missing_type += 1
                continue
            event_type = unknown_event_type

        player_name = get_player_name(event)
        if player_name:
            player_names.setdefault(player_id, player_name)

        player_ids_seen.add(player_id)
        event_types_seen.add(event_type)

        batch_feat_ids.append(build_feat_ids(event, feature_vocab))
        batch_freeze_frames.append(get_freeze_frame(event))
        batch_actor_locs.append(get_actor_loc(event))
        batch_keys.append((player_id, event_type))

        if len(batch_feat_ids) >= batch_size:
            encoded_rows += flush_embedding_batch(
                model=model,
                device=device,
                batch_feat_ids=batch_feat_ids,
                batch_freeze_frames=batch_freeze_frames,
                batch_actor_locs=batch_actor_locs,
                batch_keys=batch_keys,
                sum_vecs=sum_vecs,
                counts=counts,
            )

        if log_every > 0 and total_rows % log_every == 0:
            print(
                f"rows={total_rows} encoded={encoded_rows} "
                f"active_profiles={len(sum_vecs)} skipped_no_player={skipped_missing_player} "
                f"skipped_no_type={skipped_missing_type}",
                flush=True,
            )

    encoded_rows += flush_embedding_batch(
        model=model,
        device=device,
        batch_feat_ids=batch_feat_ids,
        batch_freeze_frames=batch_freeze_frames,
        batch_actor_locs=batch_actor_locs,
        batch_keys=batch_keys,
        sum_vecs=sum_vecs,
        counts=counts,
    )

    profiles: dict[str, dict[str, torch.Tensor]] = {}
    profile_counts: dict[str, dict[str, int]] = {}
    stored_event_types: set[str] = set()

    for (player_id, event_type), sum_vec in sum_vecs.items():
        count = counts[(player_id, event_type)]
        if count < min_count_to_store:
            continue
        profiles.setdefault(player_id, {})[event_type] = (sum_vec / count).float()
        profile_counts.setdefault(player_id, {})[event_type] = count
        stored_event_types.add(event_type)

    stored_player_ids = sorted(profiles.keys(), key=mixed_sort_key)
    stored_event_types_list = sorted(stored_event_types)
    stored_player_names = {player_id: player_names.get(player_id, player_id) for player_id in stored_player_ids}

    cache = {
        "version": 1,
        "data_path": str(data_path),
        "checkpoint_path": str(checkpoint_path),
        "device_used": str(device),
        "batch_size": batch_size,
        "min_count_to_store": min_count_to_store,
        "player_ids": stored_player_ids,
        "event_types": stored_event_types_list,
        "profiles": profiles,
        "counts": profile_counts,
        "player_names": stored_player_names,
        "stats": {
            "total_rows": total_rows,
            "encoded_rows": encoded_rows,
            "skipped_missing_player": skipped_missing_player,
            "skipped_missing_type": skipped_missing_type,
            "players_seen": len(player_ids_seen),
            "event_types_seen": len(event_types_seen),
            "stored_players": len(stored_player_ids),
            "stored_event_types": len(stored_event_types_list),
            "stored_profiles": sum(len(type_map) for type_map in profiles.values()),
        },
    }
    save_profile_cache(output_path, cache)
    return cache


def resolve_player_id_from_query(player_query: str, player_names: dict[str, str], valid_player_ids: set[str]) -> str:
    player_query = str(player_query).strip()
    if player_query in valid_player_ids:
        return player_query

    exact = [player_id for player_id, name in player_names.items() if name == player_query]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Ambiguous player name '{player_query}'. Matches ids: {', '.join(exact)}")

    folded_query = normalize_text(player_query)
    folded = [player_id for player_id, name in player_names.items() if normalize_text(name) == folded_query]
    if len(folded) == 1:
        return folded[0]
    if len(folded) > 1:
        raise ValueError(f"Ambiguous player name '{player_query}'. Matches ids: {', '.join(folded)}")

    raise ValueError(f"Player '{player_query}' not found.")


def resolve_event_type_from_query(event_type_query: str, event_types: list[str]) -> str:
    if event_type_query in event_types:
        return event_type_query

    folded_query = normalize_text(event_type_query)
    folded = [event_type for event_type in event_types if normalize_text(event_type) == folded_query]
    if len(folded) == 1:
        return folded[0]
    if len(folded) > 1:
        raise ValueError(f"Ambiguous event type '{event_type_query}'. Matches: {', '.join(folded)}")
    raise ValueError(f"Event type '{event_type_query}' not found.")


def query_player_event_type(
    cache: dict[str, Any],
    *,
    player_query: str,
    event_type_query: str,
    top_k: int,
    min_count: int,
) -> dict[str, Any]:
    profiles = cache["profiles"]
    counts = cache["counts"]
    player_names = cache.get("player_names", {})

    player_id = resolve_player_id_from_query(player_query, player_names, set(cache["player_ids"]))
    event_type = resolve_event_type_from_query(event_type_query, cache["event_types"])

    target_count = counts.get(player_id, {}).get(event_type, 0)
    target_vec = profiles.get(player_id, {}).get(event_type)
    if target_vec is None:
        raise ValueError(f"Player '{player_query}' has no stored profile for event type '{event_type}'.")
    if target_count < min_count:
        raise ValueError(
            f"Target player only has count={target_count} for event type '{event_type}', below min_count={min_count}."
        )

    candidate_ids: list[str] = []
    candidate_counts: list[int] = []
    candidate_vecs: list[torch.Tensor] = []
    for candidate_id in cache["player_ids"]:
        count = counts.get(candidate_id, {}).get(event_type, 0)
        vec = profiles.get(candidate_id, {}).get(event_type)
        if vec is None or count < min_count:
            continue
        candidate_ids.append(candidate_id)
        candidate_counts.append(count)
        candidate_vecs.append(vec.float())

    if not candidate_vecs:
        raise ValueError(f"No candidates satisfy min_count={min_count} for event type '{event_type}'.")

    matrix = torch.stack(candidate_vecs, dim=0)
    sims = cosine_scores(matrix, target_vec.float())
    ranked = torch.argsort(sims, descending=True).tolist()

    results = []
    for idx in ranked:
        candidate_id = candidate_ids[idx]
        if candidate_id == player_id:
            continue
        results.append(
            {
                "player_id": candidate_id,
                "player_name": player_names.get(candidate_id, candidate_id),
                "similarity": float(sims[idx]),
                "count": candidate_counts[idx],
            }
        )
        if len(results) >= top_k:
            break

    return {
        "player_id": player_id,
        "player_name": player_names.get(player_id, player_id),
        "event_type": event_type,
        "query_count": target_count,
        "top_k": top_k,
        "min_count": min_count,
        "results": results,
    }


def query_all_event_types(
    cache: dict[str, Any],
    *,
    player_query: str,
    top_k: int,
    min_count: int,
) -> dict[str, Any]:
    player_names = cache.get("player_names", {})
    player_id = resolve_player_id_from_query(player_query, player_names, set(cache["player_ids"]))
    player_type_counts = cache["counts"].get(player_id, {})

    event_types = sorted(
        [event_type for event_type, count in player_type_counts.items() if count >= min_count],
        key=lambda event_type: (-player_type_counts[event_type], event_type),
    )
    reports = [
        query_player_event_type(
            cache,
            player_query=player_id,
            event_type_query=event_type,
            top_k=top_k,
            min_count=min_count,
        )
        for event_type in event_types
    ]

    return {
        "player_id": player_id,
        "player_name": player_names.get(player_id, player_id),
        "min_count": min_count,
        "reports": reports,
    }


def load_global_player_embedding_assets(
    embeddings_path: Path,
    names_path: Path | None = None,
) -> tuple[list[str], torch.Tensor, dict[str, str]]:
    emb_data = torch.load(embeddings_path, map_location="cpu")
    player_ids = [str(player_id) for player_id in emb_data["player_ids"]]
    embeddings = emb_data["embeddings"].float()

    player_names: dict[str, str] = {}
    if names_path is not None and names_path.exists():
        with names_path.open("r", encoding="utf-8") as handle:
            raw_names = json.load(handle)
        player_names = {str(player_id): str(name) for player_id, name in raw_names.items()}
    elif "player_names" in emb_data:
        player_names = {str(player_id): str(name) for player_id, name in emb_data["player_names"].items()}

    return player_ids, embeddings, player_names


def find_similar_players(
    player_query: str,
    top_k: int = 10,
    embeddings_path: Path = DEFAULT_PLAYER_EMBEDDINGS,
    names_path: Path = DEFAULT_PLAYER_EMBEDDING_NAMES,
) -> list[dict[str, Any]]:
    player_ids, embeddings, player_names = load_global_player_embedding_assets(embeddings_path, names_path)
    player_id = resolve_player_id_from_query(player_query, player_names, set(player_ids))
    id_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(player_ids)}

    query_idx = id_to_idx[player_id]
    query_vec = embeddings[query_idx]
    sims = cosine_scores(embeddings, query_vec)
    ranked = torch.argsort(sims, descending=True).tolist()

    results = []
    for idx in ranked:
        candidate_id = player_ids[idx]
        if candidate_id == player_id:
            continue
        results.append(
            {
                "player_id": candidate_id,
                "player_name": player_names.get(candidate_id, candidate_id),
                "similarity": float(sims[idx]),
            }
        )
        if len(results) >= top_k:
            break
    return results


def find_similar_players_by_event_type(
    player_query: str,
    event_type: str,
    top_k: int = 10,
    min_count: int = 20,
    profiles_path: Path = DEFAULT_PROFILE_CACHE,
) -> dict[str, Any]:
    cache = load_profile_cache(profiles_path)
    return query_player_event_type(
        cache,
        player_query=player_query,
        event_type_query=event_type,
        top_k=top_k,
        min_count=min_count,
    )


def report_similarities_all_types(
    player_query: str,
    top_k: int = 10,
    min_count: int = 20,
    profiles_path: Path = DEFAULT_PROFILE_CACHE,
) -> dict[str, Any]:
    cache = load_profile_cache(profiles_path)
    return query_all_event_types(
        cache,
        player_query=player_query,
        top_k=top_k,
        min_count=min_count,
    )


def print_event_type_query(result: dict[str, Any]) -> None:
    print(
        f"Top {result['top_k']} similar players to {result['player_name']} "
        f"for event_type={result['event_type']} (query_count={result['query_count']}):"
    )
    for row in result["results"]:
        print(
            f"  {row['player_name']} "
            f"(player_id={row['player_id']}, sim={row['similarity']:.4f}, count={row['count']})"
        )


def print_all_type_report(report: dict[str, Any]) -> None:
    print(
        f"Per-event-type similarity report for {report['player_name']} "
        f"(player_id={report['player_id']}, min_count={report['min_count']}):"
    )
    for item in report["reports"]:
        print()
        print_event_type_query(item)


def cmd_build(args: argparse.Namespace) -> None:
    cache = build_player_eventtype_profiles(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        output_path=args.out,
        batch_size=args.batch_size,
        min_count_to_store=args.min_count_to_store,
        device_name=args.device,
        unknown_event_type=args.unknown_event_type,
        log_every=args.log_every,
    )
    print("Build complete.")
    print(f"Saved profile cache: {args.out}")
    print(f"Stored players: {cache['stats']['stored_players']}")
    print(f"Stored event types: {cache['stats']['stored_event_types']}")
    print(f"Stored player x event_type profiles: {cache['stats']['stored_profiles']}")
    print(f"Encoded rows: {cache['stats']['encoded_rows']}")


def cmd_query(args: argparse.Namespace) -> None:
    cache = load_profile_cache(args.profiles)
    result = query_player_event_type(
        cache,
        player_query=args.player,
        event_type_query=args.event_type,
        top_k=args.top_k,
        min_count=args.min_count,
    )
    print_event_type_query(result)


def cmd_query_all_types(args: argparse.Namespace) -> None:
    cache = load_profile_cache(args.profiles)
    report = query_all_event_types(
        cache,
        player_query=args.player,
        top_k=args.top_k,
        min_count=args.min_count,
    )
    print_all_type_report(report)


def cmd_query_global(args: argparse.Namespace) -> None:
    results = find_similar_players(
        player_query=args.player,
        top_k=args.top_k,
        embeddings_path=args.embeddings,
        names_path=args.names,
    )
    print(f"Top {args.top_k} global-similarity matches for {args.player}:")
    for row in results:
        print(f"  {row['player_name']} (player_id={row['player_id']}, sim={row['similarity']:.4f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query Player x EventType similarity profiles from EventEncoder embeddings."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Stream JSONL and build Player x EventType profile cache.")
    p_build.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p_build.add_argument("--checkpoint", type=Path, default=DEFAULT_EVENT_ENCODER_CKPT)
    p_build.add_argument("--out", type=Path, default=DEFAULT_PROFILE_CACHE)
    p_build.add_argument("--batch-size", type=int, default=256)
    p_build.add_argument("--min-count-to-store", type=int, default=1)
    p_build.add_argument("--unknown-event-type", type=str, default=None)
    p_build.add_argument("--device", type=str, default=None, help="Auto, cpu, cuda, cuda:0, etc.")
    p_build.add_argument("--log-every", type=int, default=50000)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query similar players for one event type.")
    p_query.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_CACHE)
    p_query.add_argument("--player", required=True, help="Player name or player id")
    p_query.add_argument("--event-type", required=True, help='Example: "Pass"')
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.add_argument("--min-count", type=int, default=20)
    p_query.set_defaults(func=cmd_query)

    p_query_all = sub.add_parser("query-all-types", help="Run top-k similarity for every event type of a player.")
    p_query_all.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_CACHE)
    p_query_all.add_argument("--player", required=True, help="Player name or player id")
    p_query_all.add_argument("--top-k", type=int, default=10)
    p_query_all.add_argument("--min-count", type=int, default=20)
    p_query_all.set_defaults(func=cmd_query_all_types)

    p_query_global = sub.add_parser("query-global", help="Query the original whole-player embedding cache.")
    p_query_global.add_argument("--player", required=True, help="Player name or player id")
    p_query_global.add_argument("--top-k", type=int, default=10)
    p_query_global.add_argument("--embeddings", type=Path, default=DEFAULT_PLAYER_EMBEDDINGS)
    p_query_global.add_argument("--names", type=Path, default=DEFAULT_PLAYER_EMBEDDING_NAMES)
    p_query_global.set_defaults(func=cmd_query_global)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
