"""Multi-step spiking blocks used by the end-to-end TurboVLA path.

Temporal tensors use the layout ``[T, B, N, D]``. The default attention path
is hardware-oriented Spike2Max: binary Q/K dot products are converted to
integer relative exponents, evaluated as powers of two, and normalized by a
power-of-two row denominator. The forward pass is therefore compatible with a
popcount/shift/add implementation, while straight-through estimators keep the
training gradients useful. An exact normalization mode is retained for
ablations.

Distribution-aware Q/K/V gains remain continuous during normal training. This
is important for optimization quality: the gains add no deployment operation
because they can be folded into the affine parameters of the immediately
preceding LayerNorm. Optional power-of-two QAT is exposed for targets that also
quantize those affine parameters.

The blocks use membrane shortcuts: a block receives and returns continuous
membrane/current tensors. Modality stems first project signed continuous
features and emit spikes only in their hidden branch; transformer branches use
LIF -> Linear -> Norm. Residual sums remain membrane values between blocks.
"""

from __future__ import annotations

import math
import numpy as np
import torch

# SpikingJelly 0.0.0.0.14's CuPy/AutoCUDA backend still checks ``np.int``.
# NumPy removed that deprecated alias in 1.24.  Keep the compatibility shim
# local to the spiking backend instead of downgrading NumPy or modifying the
# installed SpikingJelly package on every training machine.
if "int" not in np.__dict__:
    setattr(np, "int", int)

from spikingjelly.activation_based import neuron, surrogate
from torch import nn


class TrackedLIFNode(neuron.LIFNode):
    """Multi-step LIF node with opt-in differentiable activity tracking."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.track_activity = False
        self.last_firing_rate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes = super().forward(x)
        self.last_firing_rate = spikes.float().mean() if self.track_activity else None
        return spikes

    def reset(self) -> None:
        super().reset()
        self.last_firing_rate = None


def make_lif(tau: float, backend: str) -> TrackedLIFNode:
    return TrackedLIFNode(tau=tau, detach_reset=True, backend=backend, step_mode="m")


SPIKE_DIAGNOSTIC_NAMES = (
    "lif_firing_rate_mean",
    "lif_firing_rate_min",
    "lif_firing_rate_max",
    "qkv_firing_rate_mean",
    "attention_entropy_mean",
    "attention_top1_mass_mean",
    "active_lif_nodes",
)


def configure_lif_nodes_(
    module: nn.Module,
    threshold: float = 1.0,
    surrogate_alpha: float = 4.0,
    detach_reset: bool = True,
) -> nn.Module:
    """Apply one explicit downstream LIF policy without touching backbones."""

    if threshold <= 0.0:
        raise ValueError("LIF threshold must be positive")
    if surrogate_alpha <= 0.0:
        raise ValueError("surrogate alpha must be positive")
    for child in module.modules():
        if isinstance(child, TrackedLIFNode):
            child.v_threshold = float(threshold)
            child.detach_reset = bool(detach_reset)
            child.surrogate_function = surrogate.Sigmoid(alpha=float(surrogate_alpha))
    return module


def set_spike_diagnostics_(module: nn.Module, enabled: bool = True) -> nn.Module:
    """Enable activity and attention tracking for subsequent forwards."""

    enabled = bool(enabled)
    for child in module.modules():
        if isinstance(child, TrackedLIFNode):
            child.track_activity = enabled
            if not enabled:
                child.last_firing_rate = None
        if isinstance(child, (SpikeMultiheadAttention, SpikeBiMultiheadAttention)):
            child.collect_diagnostics = enabled
            if not enabled:
                child.last_diagnostics = {}
    return module


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def spike_activity_regularization(module: nn.Module, target_rate: float) -> torch.Tensor:
    """Penalize all tracked downstream LIF rates away from a target rate."""

    if not 0.0 <= target_rate <= 1.0:
        raise ValueError("target firing rate must be in [0, 1]")
    rates = [
        child.last_firing_rate
        for child in module.modules()
        if isinstance(child, TrackedLIFNode) and child.last_firing_rate is not None
    ]
    if not rates:
        return torch.zeros((), device=_module_device(module), dtype=torch.float32)
    stacked = torch.stack([rate.float() for rate in rates])
    return (stacked - float(target_rate)).square().mean()


@torch.no_grad()
def spike_diagnostics_vector(module: nn.Module) -> torch.Tensor:
    """Return a compact fixed-layout diagnostic vector for logging/DDP reduce."""

    device = _module_device(module)
    lif_rates = []
    qkv_rates = []
    entropies = []
    top1_masses = []
    for child in module.modules():
        if isinstance(child, TrackedLIFNode) and child.last_firing_rate is not None:
            lif_rates.append(child.last_firing_rate.detach().float())
        diagnostics = getattr(child, "last_diagnostics", None)
        if not diagnostics:
            continue
        for name, value in diagnostics.items():
            tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
            if name.endswith("firing_rate"):
                qkv_rates.append(tensor)
            elif "entropy" in name:
                entropies.append(tensor)
            elif "top1_mass" in name:
                top1_masses.append(tensor)

    def mean_or_zero(values):
        return torch.stack(values).mean() if values else torch.zeros((), device=device)

    if lif_rates:
        lif = torch.stack(lif_rates)
        lif_mean, lif_min, lif_max = lif.mean(), lif.min(), lif.max()
        active_lif_nodes = (lif > 0.0).float().sum()
    else:
        zero = torch.zeros((), device=device)
        lif_mean = lif_min = lif_max = active_lif_nodes = zero
    return torch.stack((
        lif_mean,
        lif_min,
        lif_max,
        mean_or_zero(qkv_rates),
        mean_or_zero(entropies),
        mean_or_zero(top1_masses),
        active_lif_nodes,
    ))


def clear_spike_activity_(module: nn.Module) -> None:
    """Release tracked activation graphs after a micro-batch backward."""

    for child in module.modules():
        if isinstance(child, TrackedLIFNode):
            child.last_firing_rate = None


def snapshot_spiking_state(module: nn.Module) -> dict[int, object]:
    """Clone downstream LIF membrane states for a non-invasive evaluation probe."""

    snapshot = {}
    for child in module.modules():
        if isinstance(child, TrackedLIFNode):
            value = child.v
            snapshot[id(child)] = value.detach().clone() if torch.is_tensor(value) else value
    return snapshot


@torch.no_grad()
def restore_spiking_state_(module: nn.Module, snapshot: dict[int, object]) -> None:
    """Restore a state captured by snapshot_spiking_state."""

    for child in module.modules():
        if not isinstance(child, TrackedLIFNode) or id(child) not in snapshot:
            continue
        value = snapshot[id(child)]
        child.v = value
        child.last_firing_rate = None


@torch.no_grad()
def spike_membrane_statistics(module: nn.Module) -> torch.Tensor:
    """Return mean/max absolute membrane and the active-node fraction."""

    device = _module_device(module)
    values = []
    for child in module.modules():
        if not isinstance(child, TrackedLIFNode):
            continue
        value = child.v
        if torch.is_tensor(value) and value.numel() > 0:
            values.append(value.detach().float().abs().mean())
    if not values:
        return torch.zeros(3, device=device, dtype=torch.float32)
    stacked = torch.stack(values)
    return torch.stack((
        stacked.mean(),
        stacked.max(),
        (stacked > 1.0e-6).float().mean(),
    ))


def repeat_time(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"cannot add a temporal dimension to shape {tuple(x.shape)}")
    return x.unsqueeze(0).expand(time_steps, *x.shape).contiguous()


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass and use the identity gradient backward."""

    return x + (torch.round(x) - x).detach()


def ste_floor(x: torch.Tensor) -> torch.Tensor:
    """Floor in the forward pass and use the identity gradient backward."""

    return x + (torch.floor(x) - x).detach()


def ste_clamp(x: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    """Clamp in the forward pass while retaining an identity backward path."""

    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    clamped = x.clamp(min=float(minimum), max=float(maximum))
    return x + (clamped - x).detach()


def positive_pow2_ste(
    x: torch.Tensor,
    min_shift: int,
    max_shift: int,
) -> torch.Tensor:
    """Quantize a positive gain to ``2 ** integer`` with an STE gradient."""

    if min_shift > max_shift:
        raise ValueError("min_shift must not exceed max_shift")
    lower = math.ldexp(1.0, int(min_shift))
    upper = math.ldexp(1.0, int(max_shift))
    log2_gain = torch.log2(ste_clamp(x, lower, upper))
    shift = ste_round(log2_gain).clamp(min=float(min_shift), max=float(max_shift))
    return torch.exp2(shift)


def positive_continuous_gain(
    x: torch.Tensor,
    min_gain: float = 1.0 / 16.0,
    max_gain: float = 16.0,
) -> torch.Tensor:
    """Return a positive continuous gain while preserving useful gradients.

    Distribution-aware scaling is most effective when it is trained in a
    continuous space.  Unlike an explicit runtime multiplier, this gain can be
    fused into an adjacent affine transform for deployment.
    """

    if min_gain <= 0.0 or min_gain > max_gain:
        raise ValueError("gain bounds must satisfy 0 < min_gain <= max_gain")
    return ste_clamp(x, float(min_gain), float(max_gain))


@torch.no_grad()
def fuse_gain_into_layer_norm_(
    norm: nn.LayerNorm,
    gain_parameter: nn.Parameter,
    quantize_to_pow2: bool = False,
    min_shift: int = -4,
    max_shift: int = 4,
) -> None:
    """Fold ``gain * LayerNorm(x)`` into LayerNorm's affine parameters.

    This is the exact fusion for the topology used in this file.  Folding the
    gain into the preceding Linear would *not* be exact because LayerNorm lies
    between the Linear and the gain.
    """

    if norm.weight is None:
        raise ValueError("distribution-aware gain fusion requires affine LayerNorm")
    if gain_parameter.shape != norm.weight.shape:
        raise ValueError(
            f"gain shape {tuple(gain_parameter.shape)} does not match LayerNorm "
            f"shape {tuple(norm.weight.shape)}"
        )
    if quantize_to_pow2:
        gain = positive_pow2_ste(gain_parameter, min_shift, max_shift)
    else:
        gain = positive_continuous_gain(
            gain_parameter,
            min_gain=math.ldexp(1.0, min_shift),
            max_gain=math.ldexp(1.0, max_shift),
        )
    norm.weight.mul_(gain)
    if norm.bias is not None:
        norm.bias.mul_(gain)
    gain_parameter.fill_(1.0)


class Spike2Max(nn.Module):
    """Integer-exponent Spike2Max with shift-compatible normalization.

    ``scores`` must be raw binary Q/K dot-product counts. For the default
    ``pow2_nearest`` mode, every nonzero forward attention coefficient is a
    power of two:

    1. the optional score divisor is ``2 ** score_shift``;
    2. the relative exponent is floored to an integer and clipped;
    3. base-2 exponentiation becomes a shift at deployment;
    4. the row denominator is rounded to the nearest power of two.

    The learnable temporal scale is applied *after* exponentiation, matching
    the role of the scale in the reference Spike2Max formulation. It is not an
    exponent temperature. Keeping these two roles separate avoids the coarse
    and often overly flat attention produced by quantizing a learned
    temperature inside the exponent.

    The power-of-two denominator changes an exactly normalized row only by one
    common scalar. Its row sum stays in ``[1/sqrt(2), sqrt(2)]`` (apart from
    ties at the boundary), and the following output LayerNorm removes most of
    that common-scale error. The forward pass uses the shift approximation;
    its backward pass follows exact row normalization through an STE.
    """

    _NORMALIZATION_MODES = {"pow2_nearest", "pow2_ceil", "exact", "none"}

    def __init__(
        self,
        time_steps: int,
        score_shift: int = 0,
        max_exponent: int = 12,
        normalization: str = "pow2_nearest",
        output_scale_shift_limit: int = 4,
        quantize_output_scale: bool = False,
        reference_offset: int = 1,
        temperature_shift_limit: int | None = None,
    ) -> None:
        super().__init__()
        # Backward-compatible keyword alias. The parameter is no longer used
        # as an exponent temperature; it only bounds the post-exp output gain.
        if temperature_shift_limit is not None:
            output_scale_shift_limit = int(temperature_shift_limit)
        self.time_steps = int(time_steps)
        self.score_shift = int(score_shift)
        self.max_exponent = int(max_exponent)
        self.normalization = str(normalization)
        self.output_scale_shift_limit = int(output_scale_shift_limit)
        self.quantize_output_scale = bool(quantize_output_scale)
        self.reference_offset = int(reference_offset)
        if self.time_steps < 1:
            raise ValueError("time_steps must be positive")
        if not 0 <= self.score_shift <= 30:
            raise ValueError("score_shift must be in [0, 30]")
        if not 1 <= self.max_exponent <= 30:
            raise ValueError("max_exponent must be in [1, 30]")
        if self.normalization not in self._NORMALIZATION_MODES:
            raise ValueError(
                f"normalization must be one of {sorted(self._NORMALIZATION_MODES)}, "
                f"got {self.normalization!r}"
            )
        if self.output_scale_shift_limit < 0:
            raise ValueError("output_scale_shift_limit must be non-negative")
        if self.reference_offset < 0:
            raise ValueError("reference_offset must be non-negative")
        # Keep the historical parameter name and shape for checkpoint
        # compatibility. It now has the same semantics as reference
        # Spike2Max: a positive post-exp2 temporal output gain.
        self.scale = nn.Parameter(torch.ones(self.time_steps))

    def _output_gain(self) -> torch.Tensor:
        limit = self.output_scale_shift_limit
        if self.quantize_output_scale:
            return positive_pow2_ste(self.scale, -limit, limit)
        return positive_continuous_gain(
            self.scale,
            min_gain=math.ldexp(1.0, -limit),
            max_gain=math.ldexp(1.0, limit),
        )

    def _normalize(self, weights: torch.Tensor) -> torch.Tensor:
        if self.normalization == "none":
            return weights

        # Accumulate in FP32 during training. At deployment, integer weights
        # ``1 << (max_exponent + exponent)`` can be summed in an integer
        # accumulator, so no floating-point log or division is required.
        work = weights.float()
        row_sum = work.sum(dim=-1, keepdim=True)
        valid_row = row_sum > 0.0
        safe_sum = torch.where(valid_row, row_sum, torch.ones_like(row_sum))
        exact = work / safe_sum

        if self.normalization == "exact":
            normalized = exact
        else:
            log2_sum = torch.log2(safe_sum)
            if self.normalization == "pow2_nearest":
                denominator_shift = torch.round(log2_sum)
            else:  # pow2_ceil: never amplifies a row
                denominator_shift = torch.ceil(log2_sum)
            denominator = torch.exp2(denominator_shift)
            shifted = work / denominator
            # Forward: power-of-two normalization. Backward: gradient of exact
            # row normalization, which is substantially easier to optimize.
            normalized = exact + (shifted - exact).detach()

        normalized = torch.where(valid_row, normalized, torch.zeros_like(normalized))
        return normalized.to(dtype=weights.dtype)

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
        if mask is None:
            relative_scores = scores - scores.max(dim=-1, keepdim=True).values
        else:
            # Avoid NaNs for a fully masked row: use a finite placeholder for
            # the maximum and force every masked coefficient to zero below.
            lowest = torch.finfo(scores.dtype).min
            masked_scores = scores.masked_fill(mask, lowest)
            row_has_key = (~mask).any(dim=-1, keepdim=True)
            row_max = masked_scores.max(dim=-1, keepdim=True).values
            row_max = torch.where(row_has_key, row_max, torch.zeros_like(row_max))
            relative_scores = torch.where(mask, torch.zeros_like(scores), scores - row_max)

        # ``relative_scores`` is an integer popcount difference. Division by
        # ``2 ** score_shift`` is an arithmetic shift. Flooring matches an
        # arithmetic right shift for negative values.
        exponent = relative_scores / float(1 << self.score_shift)
        exponent = ste_floor(exponent)
        exponent = exponent.clamp(min=-float(self.max_exponent), max=0.0)
        weights = torch.exp2(exponent - float(self.reference_offset))
        if mask is not None:
            weights = weights.masked_fill(mask, 0.0)
        weights = self._normalize(weights)
        output_gain = self._output_gain().to(dtype=weights.dtype)
        return weights * output_gain[:, None, None, None, None]


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
        score_shift: int | None = 0,
        attention_normalization: str = "pow2_nearest",
        attention_max_exponent: int = 12,
        attention_reference_offset: int = 1,
        quantize_qkv_scales: bool = False,
        quantize_attention_scale: bool = False,
        collect_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("spiking attention embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.time_steps = int(time_steps)
        self.quantize_qkv_scales = bool(quantize_qkv_scales)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self._qkv_scales_fused = False
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
        # Spike2Max relies on raw integer spike-coincidence differences to
        # recover winner-takes-all behavior. Therefore the high-performance
        # default is no score division. A power-of-two score shift remains
        # available as an explicit accuracy/energy ablation.
        if score_shift is None:
            score_shift = 0
        self.spike2max = Spike2Max(
            time_steps=time_steps,
            score_shift=score_shift,
            max_exponent=attention_max_exponent,
            normalization=attention_normalization,
            quantize_output_scale=quantize_attention_scale,
            reference_offset=attention_reference_offset,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, length, _ = x.shape
        return x.view(time, batch, length, self.num_heads, self.head_dim).transpose(2, 3)

    def _gain(self, parameter: nn.Parameter) -> torch.Tensor:
        if self.quantize_qkv_scales:
            return positive_pow2_ste(parameter, min_shift=-3, max_shift=4)
        return positive_continuous_gain(parameter, min_gain=1.0 / 8.0, max_gain=16.0)

    @torch.no_grad()
    def fuse_distribution_scales_(self, quantize_to_pow2: bool | None = None) -> None:
        """Fuse Q/K/V gains into LayerNorm affine parameters for inference."""

        if self._qkv_scales_fused:
            return
        quantize = self.quantize_qkv_scales if quantize_to_pow2 is None else bool(quantize_to_pow2)
        fuse_gain_into_layer_norm_(self.query_norm, self.q_scale, quantize, -3, 4)
        fuse_gain_into_layer_norm_(self.key_norm, self.k_scale, quantize, -3, 4)
        fuse_gain_into_layer_norm_(self.value_norm, self.v_scale, quantize, -3, 4)
        self._qkv_scales_fused = True

    @torch.no_grad()
    def _record_diagnostics(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        work = weights.float()
        denom = work.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        probs = work / denom
        entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=-1)
        max_entropy = math.log(max(2, weights.shape[-1]))
        self.last_diagnostics = {
            "q_firing_rate": q.float().mean(),
            "k_firing_rate": k.float().mean(),
            "v_firing_rate": v.float().mean(),
            "normalized_attention_entropy": (entropy / max_entropy).mean(),
            "attention_top1_mass": probs.max(dim=-1).values.mean(),
        }

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
        # Continuous distribution-aware gains preserve optimization quality.
        # Optional pow2 QAT can be enabled near the end of training; for
        # deployment, ``fuse_distribution_scales_`` removes these multiplies.
        if self._qkv_scales_fused:
            q = self.q_lif(self.query_norm(self.query(query_spikes)))
            k = self.k_lif(self.key_norm(self.key(context_spikes)))
            v = self.v_lif(self.value_norm(self.value(context_spikes)))
        else:
            q = self.q_lif(self.query_norm(self.query(query_spikes)) * self._gain(self.q_scale))
            k = self.k_lif(self.key_norm(self.key(context_spikes)) * self._gain(self.k_scale))
            v = self.v_lif(self.value_norm(self.value(context_spikes)) * self._gain(self.v_scale))
        q = self._heads(q)
        k = self._heads(k)
        v = self._heads(v)
        # Binary Q/K dot products are exact integer popcounts. Spike2Max handles
        # head-width scaling with a power-of-two divisor, so do not introduce a
        # floating ``1 / sqrt(head_dim)`` multiplication here.
        scores = torch.matmul(q, k.transpose(-1, -2))
        weights = self.spike2max(scores, context_padding_mask, attention_mask)
        if self.collect_diagnostics:
            self._record_diagnostics(q, k, v, weights)
        context = torch.matmul(weights, v).transpose(2, 3).contiguous()
        context = context.view(*context.shape[:3], self.embed_dim)
        return self.output_norm(self.output(self.dropout(context)))


class SpikeAttentionResidual(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
        cross_context_dim: int | None = None,
        attention_dim: int | None = None,
        layer_scale_init: float = 1.0e-2,
        score_shift: int = 0,
        attention_normalization: str = "pow2_nearest",
        attention_max_exponent: int = 12,
        attention_reference_offset: int = 1,
        quantize_qkv_scales: bool = False,
        quantize_attention_scale: bool = False,
        collect_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        context_dim = hidden_dim if cross_context_dim is None else cross_context_dim
        attention_dim = hidden_dim if attention_dim is None else int(attention_dim)
        self.attention = SpikeMultiheadAttention(
            query_dim=hidden_dim,
            context_dim=context_dim,
            embed_dim=attention_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            time_steps=time_steps,
            tau=tau,
            backend=backend,
            dropout=dropout,
            score_shift=score_shift,
            attention_normalization=attention_normalization,
            attention_max_exponent=attention_max_exponent,
            attention_reference_offset=attention_reference_offset,
            quantize_qkv_scales=quantize_qkv_scales,
            quantize_attention_scale=quantize_attention_scale,
            collect_diagnostics=collect_diagnostics,
        )
        self.scale = nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor | None = None,
        context_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = query_states if context_states is None else context_states
        delta = self.attention(query_states, context, context_padding_mask, attention_mask)
        return query_states + self.scale * delta


class SpikeFFNResidual(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        feedforward_dim: int,
        tau: float,
        backend: str,
        dropout: float,
        layer_scale_init: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.mlp_input_lif = make_lif(tau, backend)
        self.fc1 = nn.Linear(hidden_dim, feedforward_dim)
        self.fc1_norm = nn.LayerNorm(feedforward_dim)
        self.mlp_hidden_lif = make_lif(tau, backend)
        self.fc2 = nn.Linear(feedforward_dim, hidden_dim)
        self.fc2_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.mlp_input_lif(states)
        hidden = self.mlp_hidden_lif(self.fc1_norm(self.fc1(hidden)))
        branch = self.fc2_norm(self.fc2(self.dropout(hidden)))
        return states + self.scale * branch


class SpikeTransformerBlock(nn.Module):
    """Compatibility block composed from membrane attention and FFN residuals."""

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
        layer_scale_init: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.attention_residual = SpikeAttentionResidual(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            time_steps=time_steps,
            tau=tau,
            backend=backend,
            dropout=dropout,
            cross_context_dim=cross_context_dim,
            layer_scale_init=layer_scale_init,
        )
        self.ffn_residual = SpikeFFNResidual(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            tau=tau,
            backend=backend,
            dropout=dropout,
            layer_scale_init=layer_scale_init,
        )

    def forward(
        self,
        query_states: torch.Tensor,
        context_states: torch.Tensor | None = None,
        context_padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self.attention_residual(
            query_states, context_states, context_padding_mask, attention_mask
        )
        return self.ffn_residual(states)


class SpikeBiMultiheadAttention(nn.Module):
    """Efficient bidirectional visual-language spiking attention.

    By default each direction owns its Q/K projections:

    ``visual <- text`` uses ``Q_visual K_text^T`` and
    ``text <- visual`` uses ``Q_text K_visual^T``.

    The former shared-transpose affinity is retained as an explicit low-cost
    ablation, but it is no longer the default because tying both directions
    creates a substantial cross-modal expressivity bottleneck.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
        shared_affinity: bool = False,
        score_shift: int = 0,
        attention_normalization: str = "pow2_nearest",
        attention_max_exponent: int = 12,
        attention_reference_offset: int = 1,
        quantize_qkv_scales: bool = False,
        quantize_attention_scale: bool = False,
        collect_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("bidirectional attention_dim must be divisible by num_heads")
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.shared_affinity = bool(shared_affinity)
        self.quantize_qkv_scales = bool(quantize_qkv_scales)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self._qkv_scales_fused = False
        self.visual_input_lif = make_lif(tau, backend)
        self.text_input_lif = make_lif(tau, backend)
        self.visual_query = nn.Linear(hidden_dim, attention_dim)
        self.visual_key = nn.Linear(hidden_dim, attention_dim)
        self.text_query = nn.Linear(hidden_dim, attention_dim)
        self.text_key = nn.Linear(hidden_dim, attention_dim)
        self.visual_value = nn.Linear(hidden_dim, attention_dim)
        self.text_value = nn.Linear(hidden_dim, attention_dim)
        self.visual_query_norm = nn.LayerNorm(attention_dim)
        self.visual_key_norm = nn.LayerNorm(attention_dim)
        self.text_query_norm = nn.LayerNorm(attention_dim)
        self.text_key_norm = nn.LayerNorm(attention_dim)
        self.visual_value_norm = nn.LayerNorm(attention_dim)
        self.text_value_norm = nn.LayerNorm(attention_dim)
        self.visual_query_lif = make_lif(tau, backend)
        self.visual_key_lif = make_lif(tau, backend)
        self.text_query_lif = make_lif(tau, backend)
        self.text_key_lif = make_lif(tau, backend)
        self.visual_value_lif = make_lif(tau, backend)
        self.text_value_lif = make_lif(tau, backend)
        # Retain the historical names for the visual-query and text-key gains.
        self.query_gain = nn.Parameter(torch.full((attention_dim,), 7.0))
        self.key_gain = nn.Parameter(torch.full((attention_dim,), 7.0))
        self.text_query_gain = nn.Parameter(torch.full((attention_dim,), 7.0))
        self.visual_key_gain = nn.Parameter(torch.full((attention_dim,), 7.0))
        self.visual_value_gain = nn.Parameter(torch.full((attention_dim,), 7.0))
        self.text_value_gain = nn.Parameter(torch.full((attention_dim,), 7.0))
        self.visual_spike2max = Spike2Max(
            time_steps,
            score_shift=score_shift,
            max_exponent=attention_max_exponent,
            normalization=attention_normalization,
            quantize_output_scale=quantize_attention_scale,
            reference_offset=attention_reference_offset,
        )
        self.text_spike2max = Spike2Max(
            time_steps,
            score_shift=score_shift,
            max_exponent=attention_max_exponent,
            normalization=attention_normalization,
            quantize_output_scale=quantize_attention_scale,
            reference_offset=attention_reference_offset,
        )
        self.dropout = nn.Dropout(dropout)
        self.visual_output = nn.Linear(attention_dim, hidden_dim)
        self.text_output = nn.Linear(attention_dim, hidden_dim)
        self.visual_output_norm = nn.LayerNorm(hidden_dim)
        self.text_output_norm = nn.LayerNorm(hidden_dim)
        self.initialize_reverse_from_shared_()

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        time, batch, length, _ = x.shape
        return x.view(time, batch, length, self.num_heads, self.head_dim).transpose(2, 3)

    def _gain(self, parameter: nn.Parameter) -> torch.Tensor:
        if self.quantize_qkv_scales:
            return positive_pow2_ste(parameter, min_shift=-3, max_shift=4)
        return positive_continuous_gain(parameter, min_gain=1.0 / 8.0, max_gain=16.0)

    @torch.no_grad()
    def initialize_reverse_from_shared_(self) -> None:
        """Initialize independent reverse Q/K from the former shared affinity.

        This makes the independent formulation start from the same pairwise
        similarity as ``shared_affinity=True`` and then permits both directions
        to specialize during fine-tuning. After loading an older shared-affinity
        checkpoint with ``strict=False``, call this method once again.
        """

        self.text_query.load_state_dict(self.text_key.state_dict())
        self.visual_key.load_state_dict(self.visual_query.state_dict())
        self.text_query_norm.load_state_dict(self.text_key_norm.state_dict())
        self.visual_key_norm.load_state_dict(self.visual_query_norm.state_dict())
        self.text_query_gain.copy_(self.key_gain)
        self.visual_key_gain.copy_(self.query_gain)

    def _spike_projection(self, x, linear, norm, gain, lif):
        projected = norm(linear(x))
        if not self._qkv_scales_fused:
            projected = projected * self._gain(gain)
        return lif(projected)

    @torch.no_grad()
    def fuse_distribution_scales_(self, quantize_to_pow2: bool | None = None) -> None:
        if self._qkv_scales_fused:
            return
        quantize = self.quantize_qkv_scales if quantize_to_pow2 is None else bool(quantize_to_pow2)
        pairs = (
            (self.visual_query_norm, self.query_gain),
            (self.visual_key_norm, self.visual_key_gain),
            (self.text_query_norm, self.text_query_gain),
            (self.text_key_norm, self.key_gain),
            (self.visual_value_norm, self.visual_value_gain),
            (self.text_value_norm, self.text_value_gain),
        )
        for norm, gain in pairs:
            fuse_gain_into_layer_norm_(norm, gain, quantize, -3, 4)
        self._qkv_scales_fused = True

    @staticmethod
    @torch.no_grad()
    def _attention_stats(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        work = weights.float()
        probs = work / work.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, weights.shape[-1]))
        return entropy.mean(), probs.max(dim=-1).values.mean()

    def forward(
        self,
        visual_membrane: torch.Tensor,
        text_membrane: torch.Tensor,
        visual_padding_mask: torch.Tensor | None = None,
        text_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_spikes = self.visual_input_lif(visual_membrane)
        text_spikes = self.text_input_lif(text_membrane)
        qv = self._heads(self._spike_projection(
            visual_spikes, self.visual_query, self.visual_query_norm,
            self.query_gain, self.visual_query_lif
        ))
        kl = self._heads(self._spike_projection(
            text_spikes, self.text_key, self.text_key_norm,
            self.key_gain, self.text_key_lif
        ))
        if self.shared_affinity:
            ql = None
            kv = None
        else:
            ql = self._heads(self._spike_projection(
                text_spikes, self.text_query, self.text_query_norm,
                self.text_query_gain, self.text_query_lif
            ))
            kv = self._heads(self._spike_projection(
                visual_spikes, self.visual_key, self.visual_key_norm,
                self.visual_key_gain, self.visual_key_lif
            ))
        vv = self._heads(self._spike_projection(
            visual_spikes, self.visual_value, self.visual_value_norm,
            self.visual_value_gain, self.visual_value_lif
        ))
        vl = self._heads(self._spike_projection(
            text_spikes, self.text_value, self.text_value_norm,
            self.text_value_gain, self.text_value_lif
        ))
        visual_scores = torch.matmul(qv, kl.transpose(-1, -2))
        if self.shared_affinity:
            text_scores = visual_scores.transpose(-1, -2)
        else:
            assert ql is not None and kv is not None
            text_scores = torch.matmul(ql, kv.transpose(-1, -2))
        visual_weights = self.visual_spike2max(visual_scores, text_padding_mask)
        text_weights = self.text_spike2max(text_scores, visual_padding_mask)
        if self.collect_diagnostics:
            visual_entropy, visual_top1 = self._attention_stats(visual_weights)
            text_entropy, text_top1 = self._attention_stats(text_weights)
            diagnostics = {
                "visual_q_firing_rate": qv.float().mean(),
                "text_k_firing_rate": kl.float().mean(),
                "visual_v_firing_rate": vv.float().mean(),
                "text_v_firing_rate": vl.float().mean(),
                "visual_attention_entropy": visual_entropy,
                "visual_attention_top1_mass": visual_top1,
                "text_attention_entropy": text_entropy,
                "text_attention_top1_mass": text_top1,
            }
            if ql is not None and kv is not None:
                diagnostics["text_q_firing_rate"] = ql.float().mean()
                diagnostics["visual_k_firing_rate"] = kv.float().mean()
            self.last_diagnostics = diagnostics
        visual_context = torch.matmul(visual_weights, vl).transpose(2, 3).contiguous()
        text_context = torch.matmul(text_weights, vv).transpose(2, 3).contiguous()
        visual_context = visual_context.view(*visual_context.shape[:3], self.attention_dim)
        text_context = text_context.view(*text_context.shape[:3], self.attention_dim)
        visual_delta = self.visual_output_norm(self.visual_output(self.dropout(visual_context)))
        text_delta = self.text_output_norm(self.text_output(self.dropout(text_context)))
        return visual_delta, text_delta


class SpikeBiAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        num_heads: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
        layer_scale_init: float,
        shared_affinity: bool = False,
        score_shift: int = 0,
        attention_normalization: str = "pow2_nearest",
        attention_max_exponent: int = 12,
        attention_reference_offset: int = 1,
        quantize_qkv_scales: bool = False,
        quantize_attention_scale: bool = False,
        collect_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.attention = SpikeBiMultiheadAttention(
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            time_steps=time_steps,
            tau=tau,
            backend=backend,
            dropout=dropout,
            shared_affinity=shared_affinity,
            score_shift=score_shift,
            attention_normalization=attention_normalization,
            attention_max_exponent=attention_max_exponent,
            attention_reference_offset=attention_reference_offset,
            quantize_qkv_scales=quantize_qkv_scales,
            quantize_attention_scale=quantize_attention_scale,
            collect_diagnostics=collect_diagnostics,
        )
        self.visual_scale = nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
        self.text_scale = nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))

    def forward(
        self,
        visual_membrane,
        text_membrane,
        text_padding_mask,
        visual_padding_mask=None,
    ):
        visual_delta, text_delta = self.attention(
            visual_membrane, text_membrane, visual_padding_mask, text_padding_mask
        )
        return (
            visual_membrane + self.visual_scale * visual_delta,
            text_membrane + self.text_scale * text_delta,
        )


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
        cross_attention_dim: int = 512,
        cross_attention_heads: int = 8,
        cross_layer_scale_init: float = 1.0e-1,
        self_layer_scale_init: float = 1.0e-1,
        ffn_layer_scale_init: float = 1.0e-1,
        shared_cross_affinity: bool = False,
        cross_score_shift: int = 0,
        self_score_shift: int = 0,
        attention_normalization: str = "pow2_nearest",
        attention_max_exponent: int = 12,
        attention_reference_offset: int = 1,
        quantize_qkv_scales: bool = False,
        quantize_attention_scale: bool = False,
        enable_visual_self_attention: bool = False,
        enable_visual_ffn: bool = True,
        collect_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.enable_visual_self_attention = bool(enable_visual_self_attention)
        self.enable_visual_ffn = bool(enable_visual_ffn)
        self.bi_cross_layers = nn.ModuleList()
        self.visual_self_attention_layers = nn.ModuleList()
        self.visual_ffn_layers = nn.ModuleList()
        self.text_self_attention_layers = nn.ModuleList()
        self.text_ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.bi_cross_layers.append(SpikeBiAttentionBlock(
                hidden_dim=hidden_dim,
                attention_dim=cross_attention_dim,
                num_heads=cross_attention_heads,
                time_steps=time_steps,
                tau=tau,
                backend=backend,
                dropout=dropout,
                layer_scale_init=cross_layer_scale_init,
                shared_affinity=shared_cross_affinity,
                score_shift=cross_score_shift,
                attention_normalization=attention_normalization,
                attention_max_exponent=attention_max_exponent,
                attention_reference_offset=attention_reference_offset,
                quantize_qkv_scales=quantize_qkv_scales,
                quantize_attention_scale=quantize_attention_scale,
                collect_diagnostics=collect_diagnostics,
            ))
            if self.enable_visual_self_attention:
                self.visual_self_attention_layers.append(SpikeAttentionResidual(
                    hidden_dim=hidden_dim,
                    num_heads=max(1, num_heads // 2),
                    time_steps=time_steps,
                    tau=tau,
                    backend=backend,
                    dropout=dropout,
                    layer_scale_init=self_layer_scale_init,
                    score_shift=self_score_shift,
                    attention_normalization=attention_normalization,
                    attention_max_exponent=attention_max_exponent,
                    attention_reference_offset=attention_reference_offset,
                    quantize_qkv_scales=quantize_qkv_scales,
                    quantize_attention_scale=quantize_attention_scale,
                    collect_diagnostics=collect_diagnostics,
                ))
            if self.enable_visual_ffn:
                self.visual_ffn_layers.append(SpikeFFNResidual(
                    hidden_dim=hidden_dim,
                    feedforward_dim=feedforward_dim,
                    tau=tau,
                    backend=backend,
                    dropout=dropout,
                    layer_scale_init=ffn_layer_scale_init,
                ))
            self.text_self_attention_layers.append(SpikeAttentionResidual(
                hidden_dim=hidden_dim,
                num_heads=max(1, num_heads // 2),
                time_steps=time_steps,
                tau=tau,
                backend=backend,
                dropout=dropout,
                layer_scale_init=self_layer_scale_init,
                score_shift=self_score_shift,
                attention_normalization=attention_normalization,
                attention_max_exponent=attention_max_exponent,
                attention_reference_offset=attention_reference_offset,
                quantize_qkv_scales=quantize_qkv_scales,
                quantize_attention_scale=quantize_attention_scale,
                collect_diagnostics=collect_diagnostics,
            ))
            self.text_ffn_layers.append(SpikeFFNResidual(
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                tau=tau,
                backend=backend,
                dropout=dropout,
                layer_scale_init=ffn_layer_scale_init,
            ))

    @torch.no_grad()
    def upgrade_legacy_interaction_(self, cross_layer_scale: float = 0.1) -> None:
        """Initialize newly independent directions after loading an old model.

        Load an older shared-affinity checkpoint with ``strict=False`` first,
        then call this helper. It preserves the checkpoint's original affinity
        at initialization, enables later directional specialization, and opens
        the formerly near-closed cross-modal residual path.
        """

        for block in self.bi_cross_layers:
            block.attention.initialize_reverse_from_shared_()
            block.visual_scale.fill_(float(cross_layer_scale))
            block.text_scale.fill_(float(cross_layer_scale))

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_masks: torch.Tensor | None,
        visual_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer_index, (bi_cross, text_self_attention, text_ffn) in enumerate(zip(
            self.bi_cross_layers, self.text_self_attention_layers, self.text_ffn_layers
        )):
            visual_tokens, text_tokens = bi_cross(
                visual_tokens,
                text_tokens,
                text_key_padding_mask,
                visual_key_padding_mask,
            )
            # Full visual self-attention is optional because it reintroduces
            # O(N_visual^2) work. The token-wise visual FFN is linear in token
            # count and remains enabled by default.
            if self.enable_visual_self_attention:
                visual_tokens = self.visual_self_attention_layers[layer_index](
                    visual_tokens, None, visual_key_padding_mask, None
                )
            if self.enable_visual_ffn:
                visual_tokens = self.visual_ffn_layers[layer_index](visual_tokens)
            text_tokens = text_self_attention(
                text_tokens, None, text_key_padding_mask, text_self_attention_masks
            )
            text_tokens = text_ffn(text_tokens)
            text_tokens = text_tokens.masked_fill(
                text_key_padding_mask[None, :, :, None], 0.0
            )
            if visual_key_padding_mask is not None:
                visual_tokens = visual_tokens.masked_fill(
                    visual_key_padding_mask[None, :, :, None], 0.0
                )
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
    """Continuous action readout from the final membrane, without output spikes."""

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


class SpikeACTDecoderLayer(nn.Module):
    """Action-query self-attention, masked memory cross-attention, and one FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        time_steps: int,
        tau: float,
        backend: str,
        dropout: float,
        attention_scale_init: float,
        ffn_scale_init: float,
        score_shift: int = 0,
        attention_normalization: str = "pow2_nearest",
        attention_max_exponent: int = 12,
        attention_reference_offset: int = 1,
        quantize_qkv_scales: bool = False,
        quantize_attention_scale: bool = False,
        collect_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        attention_kwargs = dict(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            time_steps=time_steps,
            tau=tau,
            backend=backend,
            dropout=dropout,
            layer_scale_init=attention_scale_init,
            score_shift=score_shift,
            attention_normalization=attention_normalization,
            attention_max_exponent=attention_max_exponent,
            attention_reference_offset=attention_reference_offset,
            quantize_qkv_scales=quantize_qkv_scales,
            quantize_attention_scale=quantize_attention_scale,
            collect_diagnostics=collect_diagnostics,
        )
        self.self_attention = SpikeAttentionResidual(**attention_kwargs)
        self.cross_attention = SpikeAttentionResidual(**attention_kwargs)
        self.ffn = SpikeFFNResidual(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            tau=tau,
            backend=backend,
            dropout=dropout,
            layer_scale_init=ffn_scale_init,
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.self_attention(queries)
        queries = self.cross_attention(queries, memory, memory_padding_mask)
        return self.ffn(queries)


class SpikeACTDecoder(nn.Module):
    """Small action-query bottleneck with O(H^2 + HN) attention complexity."""

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
        attention_scale_init: float = 1.0e-1,
        ffn_scale_init: float = 1.0e-1,
        score_shift: int = 0,
        attention_normalization: str = "pow2_nearest",
        attention_max_exponent: int = 12,
        attention_reference_offset: int = 1,
        quantize_qkv_scales: bool = False,
        quantize_attention_scale: bool = False,
        collect_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.hidden_dim = int(hidden_dim)
        self.action_queries = nn.Embedding(horizon, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SpikeACTDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                time_steps=time_steps,
                tau=tau,
                backend=backend,
                dropout=dropout,
                attention_scale_init=attention_scale_init,
                ffn_scale_init=ffn_scale_init,
                score_shift=score_shift,
                attention_normalization=attention_normalization,
                attention_max_exponent=attention_max_exponent,
                attention_reference_offset=attention_reference_offset,
                quantize_qkv_scales=quantize_qkv_scales,
                quantize_attention_scale=quantize_attention_scale,
                collect_diagnostics=collect_diagnostics,
            ))
        self.action_readout = MembraneActionReadout(
            hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            action_dim=action_dim,
            tau=tau,
            backend=backend,
        )

    def _condition_queries(
        self,
        queries: torch.Tensor,
        query_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_condition is None:
            return queries
        time_steps, batch, horizon, hidden_dim = queries.shape
        if query_condition.shape[-1] != hidden_dim:
            raise ValueError(
                f"query condition hidden size {query_condition.shape[-1]} must be {hidden_dim}"
            )
        if query_condition.ndim == 2:
            if query_condition.shape[0] != batch:
                raise ValueError("[B,D] query condition has an incompatible batch size")
            condition = query_condition[None, :, None, :]
        elif query_condition.ndim == 3:
            if query_condition.shape[:2] != (time_steps, batch):
                raise ValueError("[T,B,D] query condition has an incompatible shape")
            condition = query_condition[:, :, None, :]
        elif query_condition.ndim == 4:
            if query_condition.shape[0] != time_steps or query_condition.shape[1] != batch:
                raise ValueError("[T,B,N,D] query condition has an incompatible shape")
            if query_condition.shape[2] not in (1, horizon):
                condition = query_condition.mean(dim=2, keepdim=True)
            else:
                condition = query_condition
        else:
            raise ValueError("query condition must be [B,D], [T,B,D], or [T,B,N,D]")
        return queries + condition.to(dtype=queries.dtype)

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
        query_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory.ndim != 4 or memory.shape[0] != self.time_steps:
            raise ValueError(f"decoder memory must be [T,B,N,D], got {tuple(memory.shape)}")
        batch = memory.shape[1]
        if memory_padding_mask is not None and memory_padding_mask.shape != (
            batch,
            memory.shape[2],
        ):
            raise ValueError(
                f"memory padding mask {tuple(memory_padding_mask.shape)} must be "
                f"{(batch, memory.shape[2])}"
            )
        queries = self.action_queries.weight[None, None].expand(
            self.time_steps, batch, -1, -1
        )
        queries = self._condition_queries(queries, query_condition)
        for layer in self.layers:
            queries = layer(queries, memory, memory_padding_mask)
        return self.action_readout(queries)


def set_hardware_qat_(module: nn.Module, enabled: bool = True) -> nn.Module:
    """Toggle power-of-two QAT without folding any trainable parameters.

    A practical schedule is continuous training first, followed by a short QAT
    stage. Calling this function does not provide a custom popcount kernel; it
    only makes the numerical forward path compatible with such a deployment.
    """

    enabled = bool(enabled)
    for child in module.modules():
        if isinstance(child, (SpikeMultiheadAttention, SpikeBiMultiheadAttention)):
            if child._qkv_scales_fused:
                raise RuntimeError("cannot enable QAT after distribution scales were fused")
            child.quantize_qkv_scales = enabled
        if isinstance(child, Spike2Max):
            child.quantize_output_scale = enabled
    return module


@torch.no_grad()
def prepare_hardware_inference_(
    module: nn.Module,
    quantize_distribution_scales: bool = False,
    quantize_attention_scales: bool = True,
) -> nn.Module:
    """Fuse deployment gains and enable shift-compatible attention scales.

    The function is intentionally irreversible for a live module. Save the
    trainable checkpoint before calling it. Real wall-clock acceleration still
    requires a backend that maps binary Q/K products to packed popcount and
    power-of-two products to shifts.
    """

    module.eval()
    for child in module.modules():
        if isinstance(child, (SpikeMultiheadAttention, SpikeBiMultiheadAttention)):
            child.fuse_distribution_scales_(
                quantize_to_pow2=quantize_distribution_scales
            )
        if isinstance(child, Spike2Max):
            child.quantize_output_scale = bool(quantize_attention_scales)
    return module