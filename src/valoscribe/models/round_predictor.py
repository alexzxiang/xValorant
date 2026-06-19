"""
Round win probability model — DeepSets or spatial Transformer (recommended).

Architecture
────────────
1. Per-player encoder (shared MLP): player_feats → h_i  (DeepSets)
2. Cross-player attention (optional Transformer, recommended):
      - Self-attention over all 10 player tokens
      - Spatial attention bias: proximity-weighted, inspired by GAT
        (academic reference: CSGO win-prob paper using GATs found that
         valuing player proximity improved curves vs DeepSets baselines)
3. Team mean-pooling (alive-weighted): atk_pool, def_pool
4. Global MLP: concat(atk_pool, def_pool, global_feats) → sigmoid

global_feats (31-dim) captures:
  - Tactical context: round#, time, score, spike state
  - Economy context: pistol/eco/force/full-buy per team
  - Team aggregates: alive, ults, rifles, health, loadout
  - Role-based alive: controllers/initiators (disproportionately valuable)
  - Spatial aggregates: centroid, spread, min cross-distance (0 if unavailable)

One model is trained per map — map identity is implicit in the checkpoint.

global_feat_dim is read from data at training time and stored in ModelConfig
so the same checkpoint file is self-describing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# ── Feature dimensions ─────────────────────────────────────────────────────────

NUM_AGENTS = 30         # ~27 agents + room for future additions; 0=unknown/padding
NUM_WEAPONS = 30        # specific weapon names (phantom, vandal, operator…); 0=unknown/padding
NUM_ROLES = 4           # duelist, initiator, controller, sentinel; 0=unknown/padding

AGENT_EMB_DIM = 8       # encodes playstyle, not just role
WEAPON_EMB_DIM = 8      # specific weapon — phantom vs vandal vs operator matters
ROLE_EMB_DIM = 4        # explicit role signal: controller != duelist != sentinel

PLAYER_CONTINUOUS_DIM = 11   # alive + health + armor + 3 abilities + ult + pos_x + pos_y + has_pos
PLAYER_FEAT_DIM = PLAYER_CONTINUOUS_DIM + AGENT_EMB_DIM + WEAPON_EMB_DIM + ROLE_EMB_DIM  # = 31

# global_feat_dim is not hardcoded here — it is read from data and stored in ModelConfig.global_feat_dim
# Default 31: 6 tactical + 2 economy + 10 team agg + 4 role alives + 9 spatial aggregates


@dataclass
class ModelConfig:
    player_hidden: int = 64
    player_layers: int = 2
    global_hidden: int = 64
    global_layers: int = 2
    dropout: float = 0.4
    use_transformer: bool = True     # Transformer recommended over DeepSets mean-pool
    n_heads: int = 2                 # 2 heads at d=64 → 32-dim per head
    n_transformer_layers: int = 1    # 1 layer sufficient at this data scale
    use_spatial_bias: bool = True    # proximity-weighted attention bias (GAT-inspired)
    global_feat_dim: int = 31        # set from data at build time via build_dataset.py


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden
            layers += [
                nn.Linear(d_in, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialTransformerLayer(nn.Module):
    """
    Single Transformer encoder layer with optional spatial attention bias.

    The bias is (B*n_heads, 10, 10) and is added to the raw attention logits
    before softmax — this is identical to additive positional bias in papers
    like ALiBi / Perceiver, and approximates a GAT edge weight when the bias
    encodes player distance.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x:         (B, 10, d_model)
        # attn_bias: (B * n_heads, 10, 10) — additive bias on attention logits, or None
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class RoundPredictor(nn.Module):
    """
    DeepSets (or Transformer) round win probability model.

    Input shapes:
        continuous_feats:  (B, 10, PLAYER_CONTINUOUS_DIM)  per-player continuous
        agent_ids:         (B, 10)   int64 agent indices
        weapon_ids:        (B, 10)   int64 specific weapon indices (phantom, vandal, …)
        role_ids:          (B, 10)   int64 role indices (1=duelist..4=sentinel, 0=unknown)
        global_feats:      (B, cfg.global_feat_dim)  continuous global state
        atk_mask:          (B, 10)   float 1=attack player, 0=defense
        alive_mask:        (B, 10)   float 1=alive, 0=dead (dead players excluded from pool)

    Output:
        logits: (B,) — raw logit for P(attack wins); apply sigmoid for probability
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        # Embeddings
        self.agent_emb = nn.Embedding(NUM_AGENTS + 1, AGENT_EMB_DIM, padding_idx=0)
        self.weapon_emb = nn.Embedding(NUM_WEAPONS + 1, WEAPON_EMB_DIM, padding_idx=0)
        self.role_emb = nn.Embedding(NUM_ROLES + 1, ROLE_EMB_DIM, padding_idx=0)

        # Per-player encoder — shared weights across all 10 players (DeepSets)
        self.player_encoder = MLP(PLAYER_FEAT_DIM, cfg.player_hidden, cfg.player_layers, cfg.dropout)

        # Cross-player Transformer with optional spatial attention bias (GAT-inspired)
        if cfg.use_transformer:
            self.transformer_layers = nn.ModuleList([
                SpatialTransformerLayer(cfg.player_hidden, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.n_transformer_layers)
            ])
            # Learned scale for spatial proximity bias; starts at 0 (pure attention, no bias)
            # Positive learned value → closer players get higher attention weight
            if cfg.use_spatial_bias:
                self.spatial_scale = nn.Parameter(torch.zeros(1))
            else:
                self.spatial_scale = None
        else:
            self.transformer_layers = None
            self.spatial_scale = None

        # Global MLP: [atk_pool | def_pool | global_feats]
        global_in_dim = cfg.player_hidden * 2 + cfg.global_feat_dim
        self.global_mlp = MLP(global_in_dim, cfg.global_hidden, cfg.global_layers, cfg.dropout)
        self.output_head = nn.Linear(cfg.global_hidden, 1)

    def forward(
        self,
        continuous_feats: torch.Tensor,    # (B, 10, PLAYER_CONTINUOUS_DIM)
        agent_ids: torch.Tensor,           # (B, 10)
        weapon_ids: torch.Tensor,          # (B, 10) specific weapon (phantom, vandal, …)
        role_ids: torch.Tensor,            # (B, 10)
        global_feats: torch.Tensor,        # (B, cfg.global_feat_dim)
        atk_mask: torch.Tensor,            # (B, 10) float
        alive_mask: torch.Tensor,          # (B, 10) float
    ) -> torch.Tensor:
        B = continuous_feats.shape[0]

        # Embed categorical features and concatenate with continuous
        agent_e = self.agent_emb(agent_ids)         # (B, 10, AGENT_EMB_DIM)
        weapon_e = self.weapon_emb(weapon_ids)       # (B, 10, WEAPON_EMB_DIM)
        role_e = self.role_emb(role_ids)             # (B, 10, ROLE_EMB_DIM)

        player_in = torch.cat([continuous_feats, agent_e, weapon_e, role_e], dim=-1)  # (B, 10, 31)

        # Shared MLP over all players
        player_h = self.player_encoder(player_in.view(B * 10, PLAYER_FEAT_DIM))  # (B*10, player_hidden)
        player_h = player_h.view(B, 10, -1)                                        # (B, 10, player_hidden)

        # Cross-player Transformer with optional spatial attention bias
        if self.transformer_layers is not None:
            # Build spatial bias from positions in continuous_feats
            # continuous_feats indices: 8=pos_x, 9=pos_y, 10=has_position
            attn_bias: Optional[torch.Tensor] = None
            if self.spatial_scale is not None:
                pos_xy = continuous_feats[:, :, 8:10]          # (B, 10, 2)
                has_p  = continuous_feats[:, :, 10]            # (B, 10)
                diff   = pos_xy.unsqueeze(2) - pos_xy.unsqueeze(1)   # (B, 10, 10, 2)
                dist   = diff.norm(dim=-1)                     # (B, 10, 10)
                # Only apply bias where BOTH players have detected positions
                pair_mask = has_p.unsqueeze(2) * has_p.unsqueeze(1)  # (B, 10, 10)
                # Negative distance * learned scale: closer → less negative → higher attention
                scale = torch.sigmoid(self.spatial_scale)      # keeps scale in [0, 1]
                spatial_b = -dist * pair_mask * scale          # (B, 10, 10)
                # Expand for heads: (B * n_heads, 10, 10)
                n_heads = self.config.n_heads
                attn_bias = spatial_b.unsqueeze(1).expand(-1, n_heads, -1, -1).reshape(
                    B * n_heads, 10, 10
                )

            for layer in self.transformer_layers:
                player_h = layer(player_h, attn_bias)

        # Mean-pool per team, weighted by alive_mask (dead players don't dilute the pool)
        atk_w = (atk_mask * alive_mask).unsqueeze(-1)           # (B, 10, 1)
        def_w = ((1 - atk_mask) * alive_mask).unsqueeze(-1)

        atk_pool = (player_h * atk_w).sum(1) / atk_w.sum(1).clamp(min=1.0)  # (B, player_hidden)
        def_pool = (player_h * def_w).sum(1) / def_w.sum(1).clamp(min=1.0)

        global_in = torch.cat([atk_pool, def_pool, global_feats], dim=-1)
        logit = self.output_head(self.global_mlp(global_in)).squeeze(-1)  # (B,)
        return logit

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        return torch.sigmoid(self.forward(*args, **kwargs))


# ── Input shapes summary ───────────────────────────────────────────────────────
#
# continuous_feats:  (B, 10, 11)  alive, health, armor, ab1/2/3, ult_charge,
#                                 ult_ready, pos_x[8], pos_y[9], has_pos[10]
#                                 NOTE: spatial bias reads indices 8-10 directly
# agent_ids:         (B, 10)      int64 agent indices
# weapon_ids:        (B, 10)      int64 specific weapon (phantom, vandal, operator…)
# role_ids:          (B, 10)      int64 role indices
# global_feats:      (B, 31)      tactical + economy + team aggregates + role alives
#                                 + spatial aggregates (centroid, spread, cross-dist)
#                                 dim read from data via ModelConfig.global_feat_dim
# atk_mask:          (B, 10)      float 1=attack player, 0=defense
# alive_mask:        (B, 10)      float 1=alive, 0=dead
