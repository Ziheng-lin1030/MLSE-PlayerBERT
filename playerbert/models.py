#!/usr/bin/env python3
"""Reusable PlayerBERT model definitions extracted from the notebooks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


EMBED_DIM = 128


class PlayerMLP(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 64, out_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetEncoder(nn.Module):
    def __init__(self, player_dim: int = 6, hidden: int = 64, out_dim: int = EMBED_DIM):
        super().__init__()
        self.player_mlp = PlayerMLP(in_dim=player_dim, hidden=hidden, out_dim=out_dim)

    def forward(
        self,
        freeze_frames: list[Any],
        actor_locs: list[tuple[float, float]],
        device: torch.device,
    ) -> torch.Tensor:
        batch_embeds: list[torch.Tensor] = []
        for ff, (ax, ay) in zip(freeze_frames, actor_locs):
            if ff is None or (hasattr(ff, "__len__") and len(ff) == 0):
                batch_embeds.append(torch.zeros(EMBED_DIM, device=device))
                continue

            per_player: list[torch.Tensor] = []
            for p in ff:
                if not isinstance(p, dict):
                    continue
                loc = p.get("location")
                if not isinstance(loc, list) or len(loc) < 2:
                    continue

                dx = float(loc[0]) - ax
                dy = float(loc[1]) - ay
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
                batch_embeds.append(torch.zeros(EMBED_DIM, device=device))
                continue

            players = torch.stack(per_player, dim=0)
            emb = self.player_mlp(players).mean(dim=0)
            batch_embeds.append(emb)

        return torch.stack(batch_embeds, dim=0)


class EventTransformer(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], d_model: int = EMBED_DIM, nhead: int = 4, num_layers: int = 2):
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
        tokens: list[torch.Tensor] = []
        for i, feat_name in enumerate(self.features):
            value_embed = self.value_embeds[self.name_map[feat_name]](feat_ids[:, i])
            feat_embed = self.feature_embeds.weight[i]
            tokens.append(value_embed + feat_embed.unsqueeze(0))
        x = torch.stack(tokens, dim=1)
        h = self.encoder(x)
        z_event = h.mean(dim=1)
        return z_event, h


class EventEncoder(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int]):
        super().__init__()
        self.event_encoder = EventTransformer(vocab_sizes)
        self.frame_encoder = SetEncoder()
        self.gate = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, EMBED_DIM),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_ids: torch.Tensor,
        freeze_frames: list[Any],
        actor_locs: list[tuple[float, float]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_event, h_tokens = self.event_encoder(feat_ids)
        z_frame = self.frame_encoder(freeze_frames, actor_locs, device)
        gate = self.gate(torch.cat([z_event, z_frame], dim=-1))
        z = gate * z_event + (1.0 - gate) * z_frame
        return z, h_tokens


class PlayerBERT(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, nhead: int = 4, num_layers: int = 2, max_len: int = 256):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_embed(pos)
        src_key_padding_mask = ~attn_mask
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


def load_event_encoder_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[EventEncoder, dict[str, dict[Any, int]], dict[str, Any]]:
    """Load the notebook-compatible EventEncoder and its feature vocab."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    feature_vocab = checkpoint["feature_vocab"]
    vocab_sizes = {feature_name: len(vocab) for feature_name, vocab in feature_vocab.items()}

    model = EventEncoder(vocab_sizes).to(device)
    model.load_state_dict(checkpoint["event_encoder"], strict=True)
    model.eval()
    return model, feature_vocab, checkpoint
