#!/usr/bin/env python3
"""Build facet-specific player similarity and event-level evidence on top of EventEncoder.

This script extends the original PlayerBERT workflow by:
1) exporting event embeddings E_i with metadata
2) aggregating facet-specific player embeddings (e.g., pass / block)
3) querying facet similarity
4) retrieving top-k nearest event pairs as evidence

It reuses the EventEncoder architecture defined in train_event_encoder.ipynb.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


UNK_TOKEN = "[UNK]"


def iter_json_objects(fp: Path):
    """Read JSONL-like files robustly (also handles concatenated JSON objects on one line)."""
    decoder = json.JSONDecoder()
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
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


def get_actor_loc(event: dict[str, Any]) -> tuple[float, float]:
    loc = event.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        try:
            return float(loc[0]), float(loc[1])
        except Exception:
            return 0.0, 0.0
    return 0.0, 0.0


def normalize_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if v is None:
        return UNK_TOKEN
    return str(v) if not isinstance(v, (str, int, float)) else v  # keep primitive values


class PlayerMLP(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetEncoder(nn.Module):
    def __init__(self, player_dim: int = 6, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.player_mlp = PlayerMLP(in_dim=player_dim, hidden=hidden, out_dim=out_dim)

    def forward(
        self,
        freeze_frames: list[Any],
        actor_locs: list[tuple[float, float]],
        device: torch.device,
    ) -> torch.Tensor:
        batch_embeds = []
        for ff, (ax, ay) in zip(freeze_frames, actor_locs):
            if not isinstance(ff, list) or len(ff) == 0:
                batch_embeds.append(torch.zeros(128, device=device))
                continue
            per_player = []
            for p in ff:
                if not isinstance(p, dict):
                    continue
                loc = p.get("location")
                if not isinstance(loc, list) or len(loc) < 2:
                    continue
                try:
                    dx = float(loc[0]) - ax
                    dy = float(loc[1]) - ay
                except Exception:
                    continue
                dist = math.sqrt(dx * dx + dy * dy)
                angle = math.atan2(dy, dx)
                is_teammate = 1.0 if p.get("teammate", False) else 0.0
                is_keeper = 1.0 if p.get("keeper", False) else 0.0
                vec = torch.tensor(
                    [dx, dy, dist, angle, is_teammate, is_keeper],
                    device=device,
                    dtype=torch.float32,
                )
                per_player.append(vec)
            if not per_player:
                batch_embeds.append(torch.zeros(128, device=device))
                continue
            players = torch.stack(per_player, dim=0)
            emb = self.player_mlp(players).mean(dim=0)
            batch_embeds.append(emb)
        return torch.stack(batch_embeds, dim=0)


class EventTransformer(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.features = list(vocab_sizes.keys())
        self.safe_names = [f"f{i}" for i in range(len(self.features))]
        self.name_map = dict(zip(self.features, self.safe_names))
        self.value_embeds = nn.ModuleDict(
            {self.name_map[f]: nn.Embedding(vocab_sizes[f], d_model) for f in self.features}
        )
        self.feature_embeds = nn.Embedding(len(self.features), d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feat_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for i, f in enumerate(self.features):
            v = self.value_embeds[self.name_map[f]](feat_ids[:, i])
            f_emb = self.feature_embeds.weight[i]
            tokens.append(v + f_emb.unsqueeze(0))
        x = torch.stack(tokens, dim=1)
        h = self.encoder(x)
        z_event = h.mean(dim=1)
        return z_event, h


class EventEncoder(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int]):
        super().__init__()
        self.event_encoder = EventTransformer(vocab_sizes)
        self.frame_encoder = SetEncoder()
        self.gate = nn.Sequential(nn.Linear(128 * 2, 128), nn.Sigmoid())

    def forward(
        self,
        feat_ids: torch.Tensor,
        freeze_frames: list[Any],
        actor_locs: list[tuple[float, float]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_event, h_tokens = self.event_encoder(feat_ids)
        z_frame = self.frame_encoder(freeze_frames, actor_locs, device)
        g = self.gate(torch.cat([z_event, z_frame], dim=-1))
        z = g * z_event + (1 - g) * z_frame
        return z, h_tokens


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = a / (a.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    b_n = b / (b.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return a_n @ b_n.T


def summarize_event(ev: dict[str, Any]) -> str:
    keys = [
        "type.name",
        "location_bucket.label",
        "pass.length_bucket",
        "pass.angle_bucket",
        "pass.end_location_bucket.label",
        "pass.outcome.name",
        "under_pressure",
        "position.name",
        "period",
        "minute",
        "second",
        "match_id",
    ]
    parts = []
    for k in keys:
        if k in ev and ev.get(k) is not None:
            parts.append(f"{k}={ev.get(k)}")
    return ", ".join(parts[:8]) if parts else "(no summary fields)"


def build_event_rows(
    data_path: Path,
    feature_vocab: dict[str, dict[Any, int]],
    model: EventEncoder,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    feature_list = list(feature_vocab.keys())
    events_iter = iter_json_objects(data_path)

    embeddings = []
    metadata = []

    feat_batch: list[list[int]] = []
    ff_batch: list[Any] = []
    actor_batch: list[tuple[float, float]] = []
    meta_batch: list[dict[str, Any]] = []

    def flush_batch():
        if not feat_batch:
            return
        feat_tensor = torch.tensor(feat_batch, dtype=torch.long, device=device)
        with torch.no_grad():
            z, _ = model(feat_tensor, ff_batch, actor_batch, device)
        embeddings.append(z.cpu())
        metadata.extend(meta_batch)
        feat_batch.clear()
        ff_batch.clear()
        actor_batch.clear()
        meta_batch.clear()

    for idx, ev in enumerate(events_iter):
        feat_ids = []
        for feat in feature_list:
            v = normalize_value(ev.get(feat, UNK_TOKEN))
            feat_ids.append(feature_vocab[feat].get(v, feature_vocab[feat].get(UNK_TOKEN, 0)))

        feat_batch.append(feat_ids)
        ff_batch.append(ev.get("freeze_frame"))
        actor_batch.append(get_actor_loc(ev))
        meta_batch.append(
            {
                "row_idx": idx,
                "player_id": ev.get("player.id"),
                "player_name": ev.get("player.name"),
                "match_id": ev.get("match_id"),
                "event_type": ev.get("type.name"),
                "period": ev.get("period"),
                "minute": ev.get("minute"),
                "second": ev.get("second"),
                "event": ev,
            }
        )
        if len(feat_batch) >= batch_size:
            flush_batch()

    flush_batch()

    if not embeddings:
        raise RuntimeError("No events were encoded. Check data_path.")

    return {"embeddings": torch.cat(embeddings, dim=0), "metadata": metadata}


def build_facet_rows(
    event_store: dict[str, Any],
    facet_field: str,
    min_events: int,
    robust: bool = False,
) -> dict[str, Any]:
    emb = event_store["embeddings"]
    meta = event_store["metadata"]
    groups: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    general_groups: dict[Any, list[int]] = defaultdict(list)

    for i, m in enumerate(meta):
        pid = m.get("player_id")
        ev = m.get("event", {})
        facet_value = ev.get(facet_field)
        if pid is None:
            continue
        general_groups[pid].append(i)
        if facet_value is not None:
            groups[(pid, facet_value)].append(i)

    rows = []

    def aggregate(x: torch.Tensor) -> torch.Tensor:
        if not robust or x.shape[0] < 5:
            return x.mean(dim=0)
        center = x.mean(dim=0, keepdim=True)
        d = ((x - center) ** 2).sum(dim=1)
        keep = max(1, int(math.ceil(0.8 * x.shape[0])))
        _, idxs = torch.topk(d, k=keep, largest=False)
        return x[idxs].mean(dim=0)

    for (pid, facet_value), idxs in groups.items():
        if len(idxs) < min_events:
            continue
        x = emb[idxs]
        rows.append(
            {
                "player_id": pid,
                "facet_field": facet_field,
                "facet_value": facet_value,
                "count": len(idxs),
                "embedding": aggregate(x),
            }
        )

    general_rows = []
    for pid, idxs in general_groups.items():
        if len(idxs) < min_events:
            continue
        general_rows.append({"player_id": pid, "count": len(idxs), "embedding": aggregate(emb[idxs])})

    return {"facet_rows": rows, "general_rows": general_rows}


def save_event_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, path)


def save_facet_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "facet_rows": [
            {**{k: v for k, v in row.items() if k != "embedding"}, "embedding": row["embedding"].cpu()}
            for row in store["facet_rows"]
        ],
        "general_rows": [
            {**{k: v for k, v in row.items() if k != "embedding"}, "embedding": row["embedding"].cpu()}
            for row in store["general_rows"]
        ],
    }
    torch.save(serializable, path)


def cmd_build(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(args.checkpoint, map_location=device)
    feature_vocab = state["feature_vocab"]
    vocab_sizes = {k: len(v) for k, v in feature_vocab.items()}

    model = EventEncoder(vocab_sizes).to(device)
    model.load_state_dict(state["event_encoder"], strict=True)
    model.eval()

    event_store = build_event_rows(
        data_path=args.data,
        feature_vocab=feature_vocab,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    save_event_store(args.event_out, event_store)

    facet_store = build_facet_rows(
        event_store=event_store,
        facet_field=args.facet_field,
        min_events=args.min_events,
        robust=args.robust_mean,
    )
    save_facet_store(args.facet_out, facet_store)

    print(f"Encoded events: {event_store['embeddings'].shape[0]}")
    print(f"Event embedding dim: {event_store['embeddings'].shape[1]}")
    print(f"Facet rows: {len(facet_store['facet_rows'])}")
    print(f"General rows: {len(facet_store['general_rows'])}")
    print(f"Saved event store: {args.event_out}")
    print(f"Saved facet store: {args.facet_out}")


def _find_player_name(event_meta: list[dict[str, Any]], player_id: Any) -> str | None:
    for m in event_meta:
        if m.get("player_id") == player_id and m.get("player_name"):
            return m["player_name"]
    return None


def _build_facet_matrix(facet_rows: list[dict[str, Any]], facet_field: str, facet_value: str, min_events: int):
    rows = [
        r
        for r in facet_rows
        if r.get("facet_field") == facet_field
        and str(r.get("facet_value")) == facet_value
        and int(r.get("count", 0)) >= min_events
    ]
    if not rows:
        return rows, None
    matrix = torch.stack([r["embedding"] for r in rows], dim=0).float()
    return rows, matrix


def _print_top_similar(
    facet_store: dict[str, Any],
    event_store: dict[str, Any],
    player_id: Any,
    facet_field: str,
    facet_value: str,
    top_k: int,
    min_events: int,
) -> None:
    rows, matrix = _build_facet_matrix(facet_store["facet_rows"], facet_field, facet_value, min_events)
    if matrix is None:
        print("No facet rows found for this facet.")
        return

    target_idx = None
    for i, r in enumerate(rows):
        if str(r.get("player_id")) == str(player_id):
            target_idx = i
            break
    if target_idx is None:
        print(f"Player {player_id} not found for facet {facet_field}={facet_value}.")
        return

    sims = cosine_similarity(matrix[target_idx : target_idx + 1], matrix).squeeze(0)
    ranked = torch.argsort(sims, descending=True).tolist()
    target_name = _find_player_name(event_store["metadata"], player_id)
    print(f"Facet similarity for player {player_id} ({target_name or 'unknown'}) on {facet_field}={facet_value}")
    shown = 0
    for idx in ranked:
        row = rows[idx]
        pid = row["player_id"]
        if str(pid) == str(player_id):
            continue
        name = _find_player_name(event_store["metadata"], pid)
        print(
            f"  player_id={pid} name={name or 'unknown'} "
            f"sim={float(sims[idx]):.4f} count={row['count']}"
        )
        shown += 1
        if shown >= top_k:
            break


def _event_indices_for_player_facet(
    event_store: dict[str, Any],
    player_id: Any,
    facet_field: str,
    facet_value: str,
) -> list[int]:
    idxs = []
    for i, m in enumerate(event_store["metadata"]):
        if str(m.get("player_id")) != str(player_id):
            continue
        ev = m.get("event", {})
        if str(ev.get(facet_field)) == facet_value:
            idxs.append(i)
    return idxs


def _print_evidence_pairs(
    event_store: dict[str, Any],
    player_a: Any,
    player_b: Any,
    facet_field: str,
    facet_value: str,
    top_k: int,
) -> None:
    idx_a = _event_indices_for_player_facet(event_store, player_a, facet_field, facet_value)
    idx_b = _event_indices_for_player_facet(event_store, player_b, facet_field, facet_value)
    if not idx_a or not idx_b:
        print("No events found for evidence retrieval on this facet.")
        return

    emb = event_store["embeddings"].float()
    a_mat = emb[idx_a]
    b_mat = emb[idx_b]
    sims = cosine_similarity(a_mat, b_mat)

    flat_scores = sims.flatten()
    k = min(top_k, flat_scores.numel())
    vals, flat_idx = torch.topk(flat_scores, k=k, largest=True)

    print(f"Top-{k} evidence event pairs for {facet_field}={facet_value}")
    used_pairs = set()
    out_count = 0
    for score, fid in zip(vals.tolist(), flat_idx.tolist()):
        ai = fid // sims.shape[1]
        bi = fid % sims.shape[1]
        global_ai = idx_a[ai]
        global_bi = idx_b[bi]
        # simple dedupe to avoid exact repeats in output
        if (global_ai, global_bi) in used_pairs:
            continue
        used_pairs.add((global_ai, global_bi))
        ev_a = event_store["metadata"][global_ai]["event"]
        ev_b = event_store["metadata"][global_bi]["event"]
        print(f"  pair {out_count + 1}: sim={score:.4f}")
        print(f"    A: {summarize_event(ev_a)}")
        print(f"    B: {summarize_event(ev_b)}")
        out_count += 1
        if out_count >= top_k:
            break


def cmd_query(args: argparse.Namespace) -> None:
    event_store = torch.load(args.event_store, map_location="cpu")
    facet_store = torch.load(args.facet_store, map_location="cpu")

    _print_top_similar(
        facet_store=facet_store,
        event_store=event_store,
        player_id=args.player_id,
        facet_field=args.facet_field,
        facet_value=args.facet_value,
        top_k=args.top_k,
        min_events=args.min_events,
    )

    if args.compare_player_id is not None:
        _print_evidence_pairs(
            event_store=event_store,
            player_a=args.player_id,
            player_b=args.compare_player_id,
            facet_field=args.facet_field,
            facet_value=args.facet_value,
            top_k=args.evidence_k,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facet similarity + evidence tools for PlayerBERT EventEncoder outputs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Encode events and build facet embeddings.")
    p_build.add_argument("--data", type=Path, required=True, help="Processed events JSONL (e.g., events360_v4.jsonl)")
    p_build.add_argument("--checkpoint", type=Path, required=True, help="Saved event encoder checkpoint (.pt)")
    p_build.add_argument("--event-out", type=Path, required=True, help="Output .pt file for event embeddings + metadata")
    p_build.add_argument("--facet-out", type=Path, required=True, help="Output .pt file for facet embeddings")
    p_build.add_argument("--facet-field", type=str, default="type.name", help="Facet field to aggregate by")
    p_build.add_argument("--batch-size", type=int, default=256)
    p_build.add_argument("--min-events", type=int, default=20)
    p_build.add_argument("--robust-mean", action="store_true", help="Use trimmed robust mean aggregation")
    p_build.add_argument("--device", type=str, default=None, help="cpu / cuda")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query facet similarity and evidence pairs.")
    p_query.add_argument("--event-store", type=Path, required=True)
    p_query.add_argument("--facet-store", type=Path, required=True)
    p_query.add_argument("--player-id", required=True)
    p_query.add_argument("--facet-field", type=str, default="type.name")
    p_query.add_argument("--facet-value", type=str, required=True, help='Example: "Pass"')
    p_query.add_argument("--top-k", type=int, default=5, help="Top similar players to print")
    p_query.add_argument("--min-events", type=int, default=20)
    p_query.add_argument("--compare-player-id", default=None, help="If set, also print evidence pairs vs this player")
    p_query.add_argument("--evidence-k", type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
