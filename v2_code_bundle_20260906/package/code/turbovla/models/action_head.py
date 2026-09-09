from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .components.utils import MLP
from .components.spiking import SpikeACTDecoder, SpikeStateProjection
from .configuration import ActionHeadConfig, SpikeConfig


class StateProjection(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(config.num_state_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.output_norm(tokens + self.position.to(device=tokens.device, dtype=tokens.dtype))


class ACTDecoder(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.horizon = int(config.horizon)
        self.action_queries = nn.Embedding(self.horizon, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.action_projection = MLP(hidden_dim, config.mlp_hidden_dim, config.action_dim, 3)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        hidden = self.decoder(tgt=queries, memory=memory)
        return torch.tanh(self.action_projection(hidden))


class TurboVLAActionHead(nn.Module):
    def __init__(
        self,
        config: ActionHeadConfig,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        spike_config: SpikeConfig | None = None,
    ) -> None:
        super().__init__()
        self.temporal_input = spike_config is not None and spike_config.downstream_type == "spiking"
        self.temporal_readout = "mean" if spike_config is None else spike_config.temporal_readout
        self.temporal_readout_logits: nn.Parameter | None = None
        if self.temporal_readout == "learned":
            if spike_config is None:
                raise ValueError("learned temporal readout requires a SpikeConfig")
            self.temporal_readout_logits = nn.Parameter(torch.zeros(int(spike_config.time_steps)))
        requested_type = config.decoder_type
        self.spiking = self.temporal_input if requested_type == "auto" else requested_type == "spiking"
        if self.spiking and not self.temporal_input:
            raise ValueError("spiking action head requires spike.downstream_type='spiking'")
        if self.spiking:
            self.state_projection = SpikeStateProjection(
                state_dim=config.state_dim,
                state_hidden_dim=config.state_hidden_dim,
                num_tokens=config.num_state_tokens,
                hidden_dim=hidden_dim,
                time_steps=spike_config.time_steps,
                tau=spike_config.tau,
                backend=spike_config.backend,
                dropout=config.dropout,
            )
            self.decoder = SpikeACTDecoder(
                horizon=config.horizon,
                action_dim=config.action_dim,
                hidden_dim=hidden_dim,
                num_heads=nheads,
                feedforward_dim=dim_feedforward,
                num_layers=config.num_layers,
                mlp_hidden_dim=config.mlp_hidden_dim,
                time_steps=spike_config.time_steps,
                tau=spike_config.tau,
                backend=spike_config.backend,
                dropout=config.dropout,
                attention_scale_init=spike_config.self_layer_scale_init,
                ffn_scale_init=spike_config.ffn_layer_scale_init,
                score_shift=spike_config.action_score_shift,
                attention_normalization=spike_config.attention_normalization,
                attention_max_exponent=spike_config.attention_max_exponent,
                attention_reference_offset=spike_config.attention_reference_offset,
                quantize_qkv_scales=spike_config.quantize_qkv_scales,
                quantize_attention_scale=spike_config.quantize_attention_scale,
                collect_diagnostics=spike_config.collect_diagnostics,
            )
        else:
            self.state_projection = StateProjection(config, hidden_dim)
            self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward)

    def _read_temporal_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(
                "temporal action readout expected [T,B,N,D], "
                f"got {tuple(tokens.shape)}"
            )
        if self.temporal_readout == "mean":
            return tokens.mean(dim=0)
        if self.temporal_readout == "last":
            return tokens[-1]
        if self.temporal_readout != "learned" or self.temporal_readout_logits is None:
            raise ValueError(f"unsupported temporal readout: {self.temporal_readout}")

        logits = self.temporal_readout_logits
        if logits.numel() != tokens.shape[0]:
            logits = F.interpolate(
                logits.reshape(1, 1, -1),
                size=tokens.shape[0],
                mode="linear",
                align_corners=True,
            ).reshape(-1)
        weights = torch.softmax(logits, dim=0).to(device=tokens.device, dtype=tokens.dtype)
        return (tokens * weights[:, None, None, None]).sum(dim=0)

    def forward(self, vision_language_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.temporal_input and not self.spiking:
            # Fusion returns continuous membrane/residual values, not binary spikes.
            vision_language_tokens = self._read_temporal_tokens(vision_language_tokens)
        state = state.to(device=vision_language_tokens.device, dtype=vision_language_tokens.dtype)
        state_tokens = self.state_projection(state)
        concat_dim = 2 if self.spiking else 1
        return self.decoder(torch.cat([vision_language_tokens, state_tokens], dim=concat_dim))
