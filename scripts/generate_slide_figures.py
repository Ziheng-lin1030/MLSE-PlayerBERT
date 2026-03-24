#!/usr/bin/env python3
"""Generate figures for slides_eventtype_similarity.tex from player_eventtype_profiles.pt."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cache_path = root / "models" / "player_eventtype_profiles.pt"
    out_dir = root / "docs" / "slides" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        print(f"Skip: {cache_path} not found")
        return

    cache = torch.load(cache_path, map_location="cpu")
    counts = cache["counts"]
    stats = cache.get("stats", {})

    per_type: dict[str, int] = defaultdict(int)
    for _pid, d in counts.items():
        for t, n in d.items():
            per_type[t] += int(n)

    # --- Figure 1: total event counts per type (horizontal bar)
    types_sorted = sorted(per_type.items(), key=lambda x: -x[1])
    labels = [t for t, _ in types_sorted]
    vals = [v for _, v in types_sorted]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="steelblue", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total events (summed over players)")
    ax.set_title("Event-type volume in profile cache (aggregated counts)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"event_type_totals.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: histogram of per-(player, type) event counts
    all_n: list[int] = []
    for _pid, d in counts.items():
        all_n.extend(int(n) for n in d.values())
    all_n_arr = np.array(all_n, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log10(all_n_arr + 1), bins=40, color="coral", edgecolor="white")
    ax.set_xlabel(r"$\log_{10}(n_{p,t} + 1)$")
    ax.set_ylabel("Number of (player, event_type) cells")
    ax.set_title("Distribution of support counts per stored profile")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"profile_count_distribution.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- JSON summary for slides / handouts
    summary = {
        "cache_path": str(cache_path),
        "stats": stats,
        "event_types": cache.get("event_types", []),
        "per_type_totals": dict(types_sorted),
        "per_type_totals_top5": types_sorted[:5],
    }
    (out_dir / "slide_figure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote figures and summary under {out_dir}")


if __name__ == "__main__":
    main()
