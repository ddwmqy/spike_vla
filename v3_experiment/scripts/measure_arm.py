#!/usr/bin/env python
"""四维测量中的 内存 / 能耗 / 延迟 三档(参数已由 arm_build_check.py 给出)。

能耗口径(**必须随结果一起声明**):
  - 逐算子按实际输入张量判定 AC/MAC:
      * 输入是二值 {0,1} 或 SDT-V3 式多级量化 {0,.25,.5,.75,1} → 脉冲算子,
        ACs = Σ|活动| × fan_in(accumulate only,无乘法)
      * 否则按 ANN 稠密算子 MACs = 输出元素数 × fan_in
    判定依据是数据本身,不是类名 —— 因为 SDT-V3 用 `@` 运算符且是多级量化脉冲。
  - 换算 @45nm CMOS 惯例:E = MACs × 4.6 pJ + ACs × 0.9 pJ。
  - **GPU 上是 dense PyTorch 模拟,脉冲不产生真实节能**;该数字是"若部署在事件驱动
    脉冲硬件上"的理论上界估计。

T 的处理:本代码库用 `repeat_time` 把时间步展开进 batch 维(`[T,B,N,D]`),所以稠密算子
计数天然已含 T 倍,脚本不再乘 T。

用法(必须在 GPU 上跑:spikingjelly 的 LIF 依赖 CUDA kernel):
  PYTHONPATH=<code>:<resources> python measure_arm.py --arms A0p,A1 --batches 1,16 \
      --out measure_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_build_check import ARMS, build_argv, assert_imported_code  # noqa: E402

# 45nm CMOS 惯例(能量/突触操作)
PJ_PER_MAC = 4.6
PJ_PER_AC = 0.9

# 多级量化脉冲的合法取值(SDT-V3 MultiSpike: Quant(x,0,4)/4)
_QUANT_LEVELS = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])


def _classify(activation: torch.Tensor) -> tuple[str, float, float]:
    """返回 (类型, 稀疏活动量 Σ|x|, 非零元素数)。类型 ∈ {binary, quantized, dense}。"""
    a = activation.detach()
    if a.numel() == 0:
        return "dense", 0.0, 0.0
    if a.is_floating_point():
        a = a.float()
    amax = float(a.abs().max())
    if amax == 0.0:
        return "binary", 0.0, 0.0
    absmean = float(a.abs().mean())
    if amax <= 1.0:
        # 只在小张量上精确判定取值集合(大张量用采样)
        sample = a.flatten()
        if sample.numel() > 200_000:  # 等距抽样:确定性,保证同权重跨臂数字逐位可复现
            step = sample.numel() // 200_000 + 1
            sample = sample[::step]
        uniq = torch.unique(sample)
        binary = bool(((uniq == 0) | (uniq == 1)).all())
        quantized = bool(
            ((uniq.unsqueeze(-1) - _QUANT_LEVELS.to(uniq.device)).abs() < 1e-6).any(-1).all()
        )
        nnz = float((a != 0).sum())
        act = float(a.abs().sum())
        if binary:
            return "binary", act, nnz
        if quantized:
            return "quantized", act, nnz
    return "dense", float("nan"), float("nan")


class OpAccounting(TorchDispatchMode):
    """统计 MACs/ACs,并按当前模块归属。

    归属:module_stack[-1] 是本算子的直接宿主模块名(由 forward hook 维护)。
    """

    def __init__(self, module_stack: list[str], trace: bool = False) -> None:
        super().__init__()
        self.stack = module_stack
        self.trace_enabled = trace
        self.per_module: dict[str, dict[str, float]] = {}
        self.records: list[dict] = []

    def _bucket(self) -> dict[str, float]:
        name = self.stack[-1] if self.stack else "?"
        return self.per_module.setdefault(
            name,
            {"macs": 0.0, "acs": 0.0, "acs_binary": 0.0, "ops": 0.0, "spike_ops": 0.0,
             "macs_dense_equiv": 0.0},
        )

    def _trace(self, kind: str, activation: torch.Tensor, out: torch.Tensor,
               fan_in: int, fan_out: int, counted: float, func: str) -> None:
        if not self.trace_enabled:
            return
        name = self.stack[-1] if self.stack else "?"
        self.records.append({
            "module": name,
            "func": func,
            "act_shape": list(activation.shape),
            "out_shape": list(out.shape),
            "kind": kind,
            "amax": round(float(activation.detach().abs().max()), 6),
            "act_sum": round(float(activation.detach().abs().sum()), 3),
            "fan_in": fan_in,
            "fan_out": fan_out,
            "counted": counted,
        })

    def _record(
        self,
        activation: torch.Tensor,
        out: torch.Tensor,
        fan_in: int,
        fan_out: int,
        func: str = "?",
    ) -> None:
        """统一口径(与算子形状约定无关):

          ANN  : MACs = 输出元素数 × fan_in
          脉冲 : ACs = Σ|活动| × fan_out   ← 每个活跃输入只做 fan_out 次累加

        两者在"活动密度 d"下等价于 ACs = d × MACs,自洽。
        """
        kind, act, nnz = _classify(activation)
        dense = float(out.numel()) * fan_in
        b = self._bucket()
        if kind in {"binary", "quantized"}:
            b["acs"] += act * fan_out
            b["acs_binary"] += nnz * fan_out      # 只数非零(保守口径)
            b["macs_dense_equiv"] += dense        # 若按 ANN 跑需要多少 MAC
            b["spike_ops"] += 1
            counted = act * fan_out
        else:
            b["macs"] += dense
            b["macs_dense_equiv"] += dense
            counted = dense
        b["ops"] += 1
        self._trace(kind, activation, out, fan_in, fan_out, counted, func)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            name = str(func)
            if name in ("aten.mm.default", "aten.bmm.default", "aten.addmm.default",
                        "aten.matmul.default"):
                a, b_ = args[0], args[1]
                if name == "aten.addmm.default":  # (bias, A, B)
                    a, b_ = args[1], args[2]
                # 取"更像脉冲"的那个操作数作为激活侧(另一个是权重)
                kind_a, _, _ = _classify(a)
                kind_b, _, _ = _classify(b_)
                act_side, w_side = (a, b_) if (kind_a in {"binary", "quantized"} or
                                               kind_b == "dense") else (b_, a)
                if not act_side.is_floating_point():
                    act_side = act_side.float()
                if not w_side.is_floating_point():
                    w_side = w_side.float()
                fan_in = int(a.shape[-1]) if a.dim() else 0
                fan_out = int(out.shape[-1]) if out.dim() else 0
                if fan_in and fan_out:
                    self._record(act_side, out, fan_in, fan_out, func=name)
            elif name == "aten.convolution.default":
                inp, weight = args[0], args[1]
                kh = int(weight.shape[2]) if weight.dim() > 2 else 1
                kw = int(weight.shape[3]) if weight.dim() > 3 else 1
                fan_in = int(weight[0].numel())            # C_in × kh × kw
                fan_out = int(weight.shape[0]) * kh * kw   # C_out × kh × kw
                self._record(inp, out, fan_in, fan_out, func=name)
        except Exception:  # noqa: BLE001 - 统计不能影响前向
            pass
        return out


def _install_tracking(model) -> list[str]:
    stack: list[str] = []
    for name, mod in model.named_modules():
        if not name:
            continue

        def pre_hook(_m, _inp, _name=name):
            stack.append(_name)

        def post_hook(_m, _inp, _out, _name=name):
            if stack and stack[-1] == _name:
                stack.pop()
            elif _name in stack:  # 容错
                stack.remove(_name)

        mod.register_forward_pre_hook(pre_hook)
        mod.register_forward_hook(post_hook)
    return stack


def make_inputs(batch: int, device: torch.device, num_views: int = 2, size: int = 224):
    images = torch.rand(batch, num_views, 3, size, size, device=device)
    state = torch.rand(batch, 8, device=device)
    instructions = ["put the black bowl on the plate"] * batch
    return images, state, instructions


def rollup(per_module: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, val in per_module.items():
        top = name.split(".")[0]
        dst = out.setdefault(top, {})
        for k, v in val.items():
            dst[k] = dst.get(k, 0.0) + v
    return out


def measure_arm(name: str, arm: dict, batches: list[int], iters: int, ckpt: str | None,
                trace: bool = False) -> dict:
    from turbovla.training.trainer import build_model_architecture
    from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats as parse_args

    argv = build_argv(arm, text_mode="frozen", kd=True)
    saved, sys.argv = sys.argv, ["measure_arm"] + argv
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    args.allow_hf_download = False
    if arm["vision"] == "dinov3":
        args.vision_pretrained_checkpoint = None
        args.vision_model_source_path = None

    model = build_model_architecture(args)
    init_source = "random-init"
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        init_source = f"ckpt={Path(ckpt).name} (missing={len(missing)}, unexpected={len(unexpected)})"
    model.eval()
    device = torch.device("cuda")
    model.to(device)

    result: dict = {"arm": name, "combination": arm, "init": init_source,
                    "batches": {}, "energy": {}, "latency": {}}

    for batch in batches:
        torch.cuda.empty_cache()
        images, state, instructions = make_inputs(batch, device)
        # ★ 先预热一次再做统计:SmoothSpike 的 msign(H1)(spikingbert_rot_inf.py:1019)是
        #   惰性缓存,首次前向含 5 步×3 个 768×768 矩阵乘的一次性常数开销(≈6.8G MAC),
        #   不预热会把它算进单次推理能耗,也会让 batch=1 与 batch=16 不可比。
        with torch.no_grad():
            model(instructions, images, state, reset_spiking=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stack = _install_tracking(model)
        acct = OpAccounting(stack, trace=trace)
        with torch.no_grad(), acct:
            actions = model(instructions, images, state, reset_spiking=True)
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        rolled = rollup(acct.per_module)
        macs = sum(v["macs"] for v in rolled.values())
        acs = sum(v["acs"] for v in rolled.values())
        acs_bin = sum(v["acs_binary"] for v in rolled.values())
        macs_dense = sum(v["macs_dense_equiv"] for v in rolled.values())
        e_actual = (macs * PJ_PER_MAC + acs * PJ_PER_AC) / 1e9
        e_dense = macs_dense * PJ_PER_MAC / 1e9
        result["batches"][str(batch)] = {
            "peak_alloc_MiB": round(peak_alloc, 1),
            "peak_reserved_MiB": round(peak_reserved, 1),
            "action_shape": list(actions.shape),
            "macs": macs,
            "acs": acs,
            "acs_binary": acs_bin,
            "macs_dense_equiv": macs_dense,
            "energy_mJ_weighted": e_actual,
            "energy_mJ_binary": (macs * PJ_PER_MAC + acs_bin * PJ_PER_AC) / 1e9,
            "energy_mJ_dense_equiv": e_dense,
            "energy_gain_x": (e_dense / e_actual) if e_actual > 0 else None,
            "per_module": rolled,
        }
        if trace:
            tf = Path(f"/tmp/measure_trace_{name}_b{batch}.json")
            tf.write_text(json.dumps(acct.records, ensure_ascii=False), encoding="utf-8")
            result["batches"][str(batch)]["trace_file"] = str(tf)
        if batch == batches[0]:
            # 延迟(batch=1 口径,含 tokenizer/预处理)
            n_warm = 3
            for _ in range(n_warm):
                with torch.no_grad():
                    model(instructions, images, state, reset_spiking=True)
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(instructions, images, state, reset_spiking=True)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
            result["latency"] = {
                "batch": batch,
                "median_ms": round(statistics.median(times), 3),
                "p90_ms": round(sorted(times)[int(0.9 * len(times)) - 1], 3),
                "min_ms": round(min(times), 3),
                "iters": iters,
                "note": "含 tokenize;GPU 上 dense 模拟,≠ 脉冲硬件延迟",
            }
        del images, state, instructions
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0p,A1,A2,A3")
    ap.add_argument("--batches", default="1,16")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ckpt", default=None, help="可选:加载已训 ckpt(否则随机初始化)")
    ap.add_argument("--ckpt-arm", default=None, help="该 ckpt 属于哪个臂(默认对所有臂尝试加载)")
    ap.add_argument("--trace", action="store_true", help="逐算子记录(诊断分类是否有误)")
    ap.add_argument("--out", default="measure_report.json")
    args = ap.parse_args()

    assert_imported_code()
    if not torch.cuda.is_available():
        raise SystemExit("[FATAL] 需要 GPU:spikingjelly 的 LIF 依赖 CUDA kernel")
    print(f"[env] GPU: {torch.cuda.get_device_name(0)}")

    batches = [int(b) for b in args.batches.split(",")]
    reports, failures = [], []
    for name in [n.strip() for n in args.arms.split(",") if n.strip()]:
        if name not in ARMS:
            failures.append({"arm": name, "error": "unknown arm"})
            continue
        print(f"\n===== {name}: {ARMS[name]} =====", flush=True)
        ckpt = args.ckpt if (args.ckpt and (args.ckpt_arm in (None, name))) else None
        try:
            rep = measure_arm(name, ARMS[name], batches, args.iters, ckpt, trace=args.trace)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failures.append({"arm": name, "error": f"{type(exc).__name__}: {exc}"})
            torch.cuda.empty_cache()
            continue
        reports.append(rep)
        for b, data in rep["batches"].items():
            gain = data["energy_gain_x"]
            print(f"  batch={b:>3}  峰值显存 {data['peak_alloc_MiB']:>8.1f} MiB | "
                  f"MACs {data['macs']/1e9:>8.3f}G | ACs {data['acs']/1e9:>8.3f}G "
                  f"(二值口径 {data['acs_binary']/1e9:.3f}G) | "
                  f"能量 {data['energy_mJ_weighted']:.2f} mJ "
                  f"(全 ANN 等价 {data['energy_mJ_dense_equiv']:.2f} mJ"
                  + (f", 增益 {gain:.2f}×)" if gain else ")"))
        if rep.get("latency"):
            lat = rep["latency"]
            print(f"  延迟 batch={lat['batch']}: 中位 {lat['median_ms']:.2f} ms "
                  f"| p90 {lat['p90_ms']:.2f} ms")
        for mod, val in sorted(rep["batches"][str(batches[0])]["per_module"].items(),
                               key=lambda kv: -(kv[1]["macs"] + kv[1]["acs"])):
            print(f"    {mod:<28} MACs {val['macs']/1e9:>7.3f}G | ACs {val['acs']/1e9:>7.3f}G "
                  f"| 脉冲算子 {int(val['spike_ops'])}/{int(val['ops'])}")

    Path(args.out).write_text(
        json.dumps({"reports": reports, "failures": failures}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n报告: {args.out}")
    if failures:
        print(f"[RESULT] FAIL — {len(failures)} 项失败")
        return 1
    print(f"[RESULT] PASS — {len(reports)} 臂测量完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
