"""Multi-step spiking blocks used by the end-to-end TurboVLA path.

Temporal tensors use the layout ``[T, B, N, D]``. Attention normalization
matches SpikingLM's Spike2Max rule: subtract the row maximum and one, clamp,
then apply base-2 exponentiation with a learnable per-timestep scale.

The blocks use membrane shortcuts: a block receives and returns continuous
membrane/current tensors. Modality stems first project signed continuous
features and emit spikes only in their hidden branch; transformer branches use
LIF -> Linear -> Norm. Residual sums remain membrane values between blocks.
"""

from __future__ import annotations

import numpy as np
import torch

# SpikingJelly 0.0.0.0.14's CuPy/AutoCUDA backend still checks ``np.int``.
# NumPy removed that deprecated alias in 1.24.  Keep the compatibility shim
# local to the spiking backend instead of downgrading NumPy or modifying the
# installed SpikingJelly package on every training machine.
if "int" not in np.__dict__:
    setattr(np, "int", int)

from spikingjelly.activation_based import neuron
from torch import nn


def make_lif(tau: float, backend: str) -> neuron.LIFNode:
    return neuron.LIFNode(tau=tau, detach_reset=True, backend=backend, step_mode="m")


def repeat_time(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"cannot add a temporal dimension to shape {tuple(x.shape)}")
    return x.unsqueeze(0).expand(time_steps, *x.shape).contiguous()


class Spike2Max(nn.Module):
    """SpikingLM-compatible exp2 attention without floating-point Softmax."""

    def __init__(self, time_steps: int) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.scale = nn.Parameter(torch.ones(self.time_steps))

    def forward(
        self,
        scores: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scores.ndim != 5 or scores.shape[0] != self.time_steps:
            raise ValueError(f"Spike2Max expected [T,B,H,Q,K], got {tuple(scores.shape)}")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (scores.shape[1], scores.shape[-1]):
                raise ValueError(
                    f"key padding mask {tuple(key_padding_mask.shape)} is incompatible with {tuple(scores.shape)}"
                )
            mask = key_padding_mask[None, :, None, None, :]
        else:
            mask = None
        if attention_mask is not None:
            expected = (scores.shape[1], scores.shape[-2], scores.shape[-1])
            if attention_mask.shape != expected:
                raise ValueError(f"attention mask {tuple(attention_mask.shape)} must be {expected}")
            blocked = ~attention_mask[None, :, None, :, :]
            mask = blocked if mask is None else (mask | blocked)
        if mask is not None:
            scores = scores.masked_fill(mask, -1.0e4)
        scores = scores - scores.max(dim=-1, keepdim=True).values - 1.0
        # Once Spike2Max is normalized, an outside multiplicative scale would
        # cancel exactly. Use it as a learnable inverse temperature instead.
        temperature = self.scale.clamp(min=0.125, max=8.0)
        scores = scores * temperature[:, None, None, None, None].to(dtype=scores.dtype)
        weights = torch.exp2(scores.clamp(min=-30.0, max=30.0))
        if mask is not None:
            weights = weights.masked_fill(mask, 0.0)
        # SpikingLM applies Spike2Max to short text sequences. TurboVLA also
        # attends over 392/415 visual-policy tokens, so the unnormalized sum
        # otherwise grows with context length. Keep exp2 spike scoring while
        # making the aggregation scale independent of Nk.
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


class SpikeLinearProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, time_steps: int, tau: float, backend: str) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.linear = nn.Linear(in_dim, out_dim)
        self.output_norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            temporal = repeat_time(x, self.time_steps)
        elif x.ndim == 4 and x.shape[0] == self.time_steps:
            temporal = x
        else:
            raise ValueError(f"SpikeLinearProjection expected [B,N,D] or [T,B,N,D], got {tuple(x.shape)}")
        # The SpikingLM output is a signed membrane tensor. Project it before
        # the first downstream LIF so negative semantic channels are retained;
        # the receiving attention block performs the spike conversion.
        return self.output_norm(self.linear(temporal))


class SpikeVisionProjection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
        time_steps: int,
        tau: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc1_norm = nn.LayerNorm(hidden_dim)
        self.hidden_lif = make_lif(tau, backend)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.fc2_norm = nn.LayerNorm(out_dim)
        self.skip_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"SpikeVisionProjection expected [B,V,N,D], got {tuple(tokens.shape)}")
        temporal = repeat_time(tokens, self.time_steps)
        # SDT-V3 forward_features is a signed continuous residual. The analog
        # stem learns a polarity-preserving projection before spike emission.
        hidden = self.hidden_lif(self.fc1_norm(self.fc1(temporal)))
        branch = self.fc2_norm(self.fc2(self.dropout(hidden)))
        shortcut = self.skip_norm(self.skip(temporal))
        return shortcut + branch


class SpikeMultiheadAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("spiking attention embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.time_steps = int(time_steps)
        self.query_input_lif = make_lif(tau, backend)
        self.context_input_lif = make_lif(tau, backend)
        self.query = nn.Linear(query_dim, embed_dim)
        self.key = nn.Linear(context_dim, embed_dim)
        self.value = nn.Linear(context_dim, embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        self.value_norm = nn.LayerNorm(embed_dim)
        self.q_scale = nn.Parameter(torch.full((embed_dim,), 7.0))
        self.k_scale = nn.Parameter(torch.full((embed_dim,), 7.0))
        self.v_scale = nn.Parameter(torch.full((embed_dim,), 7.0))
        self.q_lif = make_lif(tau, backend)
        self.k_lif = make_lif(tau, backend)
        self.v_lif = make_lif(tau, backend)
        self.spike2max = Spike2Max(time_steps)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, length, _ = x.shape
        return x.view(time, batch, length, self.num_heads, self.head_dim).transpose(2, 3)

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor,
        context_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_states.ndim != 4 or context_states.ndim != 4:
            raise ValueError("spiking attention inputs must be [T,B,N,D]")
        query_spikes = self.query_input_lif(query_states)
        context_spikes = self.context_input_lif(context_states)
        q = self.q_lif(self.query_norm(self.query(query_spikes)) * self.q_scale)
        k = self.k_lif(self.key_norm(self.key(context_spikes)) * self.k_scale)
        v = self.v_lif(self.value_norm(self.value(context_spikes)) * self.v_scale)
        q = self._heads(q)
        k = self._heads(k)
        v = self._heads(v)
        # Binary Q/K dot products still grow with head width. Scaling before
        # Spike2Max prevents nearly one-hot attention at initialization.
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        weights = self.spike2max(scores, context_padding_mask, attention_mask)
        context = torch.matmul(weights, v).transpose(2, 3).contiguous()
        context = context.view(*context.shape[:3], self.embed_dim)
        return self.output_norm(self.output(self.dropout(context)))


class SpikeTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
        cross_context_dim: int | None = None,
    ) -> None:
        super().__init__()
        context_dim = hidden_dim if cross_context_dim is None else cross_context_dim
        self.attention = SpikeMultiheadAttention(
            query_dim=hidden_dim,
            context_dim=context_dim,
            embed_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            time_steps=time_steps,
            tau=tau,
            backend=backend,
            dropout=dropout,
        )
        self.mlp_input_lif = make_lif(tau, backend)
        self.fc1 = nn.Linear(hidden_dim, feedforward_dim)
        self.fc1_norm = nn.LayerNorm(feedforward_dim)
        self.mlp_hidden_lif = make_lif(tau, backend)
        self.fc2 = nn.Linear(feedforward_dim, hidden_dim)
        self.fc2_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # LayerScale keeps deep membrane shortcuts stable at initialization;
        # both scales remain learnable and can grow with training.
        self.attention_scale = nn.Parameter(torch.full((hidden_dim,), 1.0e-1))
        self.mlp_scale = nn.Parameter(torch.full((hidden_dim,), 1.0e-1))

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor | None = None,
        context_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = query_states if context_states is None else context_states
        delta = self.attention(
            query_states,
            context,
            context_padding_mask,
            attention_mask,
        )
        states = query_states + self.attention_scale * delta
        hidden = self.mlp_input_lif(states)
        hidden = self.mlp_hidden_lif(self.fc1_norm(self.fc1(hidden)))
        branch = self.fc2_norm(self.fc2(self.dropout(hidden)))
        return states + self.mlp_scale * branch


class SpikeBiMultiheadAttention(nn.Module):
    """Yesterday's spike attention math with one shared V-L affinity matrix."""

    def __init__(self, hidden_dim, num_heads, time_steps, tau, backend, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.visual_input_lif = make_lif(tau, backend)
        self.text_input_lif = make_lif(tau, backend)
        self.visual_query = nn.Linear(hidden_dim, hidden_dim)
        self.text_key = nn.Linear(hidden_dim, hidden_dim)
        self.visual_value = nn.Linear(hidden_dim, hidden_dim)
        self.text_value = nn.Linear(hidden_dim, hidden_dim)
        self.visual_query_norm = nn.LayerNorm(hidden_dim)
        self.text_key_norm = nn.LayerNorm(hidden_dim)
        self.visual_value_norm = nn.LayerNorm(hidden_dim)
        self.text_value_norm = nn.LayerNorm(hidden_dim)
        self.visual_query_lif = make_lif(tau, backend)
        self.text_key_lif = make_lif(tau, backend)
        self.visual_value_lif = make_lif(tau, backend)
        self.text_value_lif = make_lif(tau, backend)
        self.query_scale = nn.Parameter(torch.full((hidden_dim,), 7.0))
        self.key_scale = nn.Parameter(torch.full((hidden_dim,), 7.0))
        self.visual_value_scale = nn.Parameter(torch.full((hidden_dim,), 7.0))
        self.text_value_scale = nn.Parameter(torch.full((hidden_dim,), 7.0))
        self.visual_spike2max = Spike2Max(time_steps)
        self.text_spike2max = Spike2Max(time_steps)
        self.dropout = nn.Dropout(dropout)
        self.visual_output = nn.Linear(hidden_dim, hidden_dim)
        self.text_output = nn.Linear(hidden_dim, hidden_dim)
        self.visual_output_norm = nn.LayerNorm(hidden_dim)
        self.text_output_norm = nn.LayerNorm(hidden_dim)

    def _heads(self, x):
        time, batch, length, _ = x.shape
        return x.view(time, batch, length, self.num_heads, self.head_dim).transpose(2, 3)

    def _project(self, x, linear, norm, scale, lif):
        return lif(norm(linear(x)) * scale)

    def forward(self, visual_membrane, text_membrane, text_padding_mask):
        visual_spikes = self.visual_input_lif(visual_membrane)
        text_spikes = self.text_input_lif(text_membrane)
        qv = self._heads(self._project(
            visual_spikes, self.visual_query, self.visual_query_norm,
            self.query_scale, self.visual_query_lif
        ))
        kl = self._heads(self._project(
            text_spikes, self.text_key, self.text_key_norm,
            self.key_scale, self.text_key_lif
        ))
        vv = self._heads(self._project(
            visual_spikes, self.visual_value, self.visual_value_norm,
            self.visual_value_scale, self.visual_value_lif
        ))
        vl = self._heads(self._project(
            text_spikes, self.text_value, self.text_value_norm,
            self.text_value_scale, self.text_value_lif
        ))
        shared_scores = torch.matmul(qv, kl.transpose(-1, -2)) * (self.head_dim ** -0.5)
        visual_weights = self.visual_spike2max(shared_scores, text_padding_mask)
        text_weights = self.text_spike2max(shared_scores.transpose(-1, -2), None)
        visual_context = torch.matmul(visual_weights, vl).transpose(2, 3).contiguous()
        text_context = torch.matmul(text_weights, vv).transpose(2, 3).contiguous()
        visual_context = visual_context.view(*visual_context.shape[:3], self.hidden_dim)
        text_context = text_context.view(*text_context.shape[:3], self.hidden_dim)
        visual_delta = self.visual_output_norm(self.visual_output(self.dropout(visual_context)))
        text_delta = self.text_output_norm(self.text_output(self.dropout(text_context)))
        return visual_delta, text_delta


class SpikeBiAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, time_steps, tau, backend, dropout):
        super().__init__()
        self.attention = SpikeBiMultiheadAttention(
            hidden_dim, num_heads, time_steps, tau, backend, dropout
        )
        # Match yesterday's residual update strength.
        self.visual_scale = nn.Parameter(torch.full((hidden_dim,), 1.0e-1))
        self.text_scale = nn.Parameter(torch.full((hidden_dim,), 1.0e-1))

    def forward(self, visual_membrane, text_membrane, text_padding_mask):
        visual_delta, text_delta = self.attention(
            visual_membrane, text_membrane, text_padding_mask
        )
        return (
            visual_membrane + self.visual_scale * visual_delta,
            text_membrane + self.text_scale * text_delta,
        )


class SpikeFFNResidual(nn.Module):
    """The FFN half of yesterday's SpikeTransformerBlock."""

    def __init__(self, hidden_dim, feedforward_dim, tau, backend, dropout):
        super().__init__()
        self.input_lif = make_lif(tau, backend)
        self.fc1 = nn.Linear(hidden_dim, feedforward_dim)
        self.fc1_norm = nn.LayerNorm(feedforward_dim)
        self.hidden_lif = make_lif(tau, backend)
        self.fc2 = nn.Linear(feedforward_dim, hidden_dim)
        self.fc2_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((hidden_dim,), 1.0e-1))

    def forward(self, states):
        hidden = self.input_lif(states)
        hidden = self.hidden_lif(self.fc1_norm(self.fc1(hidden)))
        branch = self.fc2_norm(self.fc2(self.dropout(hidden)))
        return states + self.scale * branch


class SpikeVisionLanguageInteraction(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bi_cross_layers = nn.ModuleList()
        self.visual_cross_ffn_layers = nn.ModuleList()
        self.text_cross_ffn_layers = nn.ModuleList()
        self.text_self_layers = nn.ModuleList()
        for _ in range(num_layers):
            common = dict(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                time_steps=time_steps,
                tau=tau,
                backend=backend,
                dropout=dropout,
            )
            self.bi_cross_layers.append(SpikeBiAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                time_steps=time_steps,
                tau=tau,
                backend=backend,
                dropout=dropout,
            ))
            self.visual_cross_ffn_layers.append(SpikeFFNResidual(
                hidden_dim, feedforward_dim, tau, backend, dropout
            ))
            self.text_cross_ffn_layers.append(SpikeFFNResidual(
                hidden_dim, feedforward_dim, tau, backend, dropout
            ))
            self.text_self_layers.append(SpikeTransformerBlock(**common))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for bi_cross_layer, visual_ffn, text_cross_ffn, text_self_layer in zip(
            self.bi_cross_layers,
            self.visual_cross_ffn_layers,
            self.text_cross_ffn_layers,
            self.text_self_layers,
        ):
            visual_tokens, text_tokens = bi_cross_layer(
                visual_tokens, text_tokens, text_key_padding_mask
            )
            visual_tokens = visual_ffn(visual_tokens)
            text_tokens = text_cross_ffn(text_tokens)
            text_tokens = text_self_layer(
                text_tokens,
                None,
                text_key_padding_mask,
                text_self_attention_masks,
            )
            text_tokens = text_tokens.masked_fill(text_key_padding_mask[None, :, :, None], 0.0)
        return visual_tokens, text_tokens


class SpikeStateProjection(nn.Module):
    def __init__(
        self,
        state_dim: int,
        state_hidden_dim: int,
        num_tokens: int,
        hidden_dim: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(state_dim, state_hidden_dim)
        self.fc1_norm = nn.LayerNorm(state_hidden_dim)
        self.hidden_lif = make_lif(tau, backend)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(state_hidden_dim, num_tokens * hidden_dim)
        self.fc2_norm = nn.LayerNorm(hidden_dim)
        self.position = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        temporal = repeat_time(state, self.time_steps)
        # Robot state is signed and low-dimensional; an input LIF would erase
        # negative directions before any learnable encoding.
        hidden = self.hidden_lif(self.fc1_norm(self.fc1(temporal)))
        tokens = self.fc2(self.dropout(hidden)).view(
            self.time_steps, state.shape[0], self.num_tokens, self.hidden_dim
        )
        return self.fc2_norm(tokens) + self.position[None]


class MembraneActionReadout(nn.Module):
    """Continuous action readout from the last membrane potential, without output spikes."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int,
        action_dim: int,
        tau: float,
        backend: str,
    ) -> None:
        super().__init__()
        self.input_lif = make_lif(tau, backend)
        self.fc1 = nn.Linear(hidden_dim, mlp_hidden_dim)
        self.fc1_norm = nn.LayerNorm(mlp_hidden_dim)
        self.hidden_lif = make_lif(tau, backend)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc2_norm = nn.LayerNorm(mlp_hidden_dim)
        self.output_lif = make_lif(tau, backend)
        self.fc3 = nn.Linear(mlp_hidden_dim, action_dim)
        self.tau = float(tau)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        spikes = self.hidden_lif(self.fc1_norm(self.fc1(self.input_lif(hidden))))
        spikes = self.output_lif(self.fc2_norm(self.fc2(spikes)))
        currents = self.fc3(spikes)
        membrane = torch.zeros_like(currents[0])
        for step in range(currents.shape[0]):
            membrane = membrane + (currents[step] - membrane) / self.tau
        return torch.tanh(membrane)


class SpikeACTDecoder(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        mlp_hidden_dim: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.action_queries = nn.Embedding(horizon, hidden_dim)
        self.self_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        for _ in range(num_layers):
            common = dict(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                time_steps=time_steps,
                tau=tau,
                backend=backend,
                dropout=dropout,
            )
            self.self_layers.append(SpikeTransformerBlock(**common))
            self.cross_layers.append(SpikeTransformerBlock(**common))
        self.action_readout = MembraneActionReadout(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            action_dim=action_dim,
            tau=tau,
            backend=backend,
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[1]
        queries = self.action_queries.weight[None, None].expand(self.time_steps, batch, -1, -1)
        for self_layer, cross_layer in zip(self.self_layers, self.cross_layers):
            queries = self_layer(queries)
            queries = cross_layer(queries, memory)
        return self.action_readout(queries)
