#!/usr/bin/env python
"""四臂模型构建校验 + 参数统计(不训练、不碰数据)。

目的:在花 13 小时训练之前,先用**真实 trainer 代码路径**验证每个臂的 flag 组合能否
正确构建出目标架构,并顺带产出消融表的"参数"一维。

做法:
  1. 用 turbo vla 自己的 `parse_args()` 解析每个臂的 argv → 校验 flag 名/取值合法;
  2. 用 `build_model_architecture(args)` 真实构建 → 校验权重能加载;
  3. 记录各子模块的类名、参数量、以及**动作头签名哈希**(证明 ACT 未随臂改变)。

用法(pod 或服务器,CPU 即可):
  PYTHONPATH=<code>:<resources> python arm_build_check.py --out arm_build_report.json

注意:本脚本不写任何 checkpoint、不建数据集、不动 GPU。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 资产路径:默认指向共享卷 v2 包(pod/服务器通用);可用环境变量覆盖。
# 与 run_train_v2.sh 的 REPO_DIR 一一对应。
# ---------------------------------------------------------------------------
RES = Path(
    os.environ.get(
        "V3_RESOURCES",
        "/data/260010028/dwh_vla/v2_code_bundle_20260906/resources",
    )
)
DINOV3_PATH = RES / "pretrained/dinov3-vitb16-pretrain-lvd1689m"
BERT_PATH = RES / "pretrained/bert-base-uncased"
SDTV3_CHECKPOINT = RES / "pretrained/V3_19.0M_1x4.pth"
SDTV3_SOURCE = (
    RES / "third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py"
)
SOOTSPIKE_WEIGHTS = RES / "pretrained/SmoothSpike/smoothspike-bert-base-fused"
SOOTSPIKE_SOURCE = RES / "third_party/SmoothSpike"
STATS_PATH = RES / "experiments/libero/configs/libero_all4_stats.json"
TEXT_LAYOUT_PATH = RES / "experiments/libero/configs/online_text_layout.json"
OFFICIAL_CKPT = RES / "pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth"

# ---------------------------------------------------------------------------
# 四臂定义:只有这三个轴不同,其余(KD/ACT/优化/数据)全部一致。
# ---------------------------------------------------------------------------
ARMS: dict[str, dict[str, str]] = {
    "A0p": {"vision": "dinov3", "text": "bert", "downstream": "ann"},      # 自训 ANN 对照(消 KD 混淆)
    "A1": {"vision": "sdtv3_19m", "text": "bert", "downstream": "ann"},    # 仅视觉脉冲
    "A2": {"vision": "dinov3", "text": "sootspike", "downstream": "ann"},  # 仅文本脉冲
    "A3": {"vision": "sdtv3_19m", "text": "sootspike", "downstream": "ann"},  # 前端全脉冲
    "A4": {"vision": "sdtv3_19m", "text": "sootspike", "downstream": "spiking"},  # 已有参考臂(复核)
}


def build_argv(arm: dict[str, str], *, text_mode: str, kd: bool) -> list[str]:
    """复刻 run_train_v2.sh 的 COMMAND 数组,只切 vision/text/downstream 三轴。"""
    dataset_dirs = ",".join(
        str(RES / f"data/libero/libero_{suite}_no_noops/1.0.0")
        for suite in ("10", "goal", "object", "spatial")
    )
    argv = [
        "--dinov3_path", str(DINOV3_PATH),
        # mixed-suite 入口(turbovla.training.train_mixed)独有的四个参数
        "--dataset_dirs", dataset_dirs,
        "--stats_path", str(STATS_PATH),
        "--stats_key", "libero_all4_no_noops",
        "--dinov3_precision", "fp32",
        "--text_layout_path", str(TEXT_LAYOUT_PATH),
        # --- vision ---
        "--vision_encoder_type", arm["vision"],
        "--vision_image_size", "224",
        "--vision_output_grid_size", "14",
        "--vision_precision", "fp32",
        "--no_freeze_backbones",
    ]
    if arm["vision"] == "sdtv3_19m":
        argv += [
            "--vision_model_path", str(SDTV3_CHECKPOINT),
            "--vision_pretrained_checkpoint", str(SDTV3_CHECKPOINT),
            "--vision_model_source_path", str(SDTV3_SOURCE),
        ]
    else:  # dinov3:权重来自 HF 本地目录,SDT-V3 的三个路径全部不适用
        argv += [
            "--vision_model_path", str(DINOV3_PATH),
            "--vision_pretrained_checkpoint", "",
            "--vision_model_source_path", "",
        ]

    # --- text ---
    argv += ["--text_encoder_type", arm["text"]]
    if arm["text"] == "bert":
        argv += ["--bert_path", str(BERT_PATH)]           # 不需要 source path
    elif arm["text"] == "sootspike":
        argv += [
            "--bert_path", str(SOOTSPIKE_WEIGHTS),
            "--text_source_path", str(SOOTSPIKE_SOURCE),
        ]
    else:
        raise ValueError(f"unsupported text encoder for this check: {arm['text']}")
    argv += ["--spikinglm_time_steps", "4"]

    # --- fusion / downstream ---
    argv += [
        "--downstream_type", arm["downstream"],
        "--downstream_time_steps", "4",
        "--downstream_lif_tau", "2.0",
        "--downstream_lif_backend", "cupy",
        "--spike_reset_policy", "per_forward",
        "--spike_attention_normalization", "exact",
        "--spike_cross_attention_dim", "512",
        "--spike_cross_attention_heads", "8",
        "--spike_cross_layer_scale_init", "0.01",
        "--spike_self_layer_scale_init", "0.1",
        "--spike_ffn_layer_scale_init", "0.1",
        "--spike_collect_diagnostics",
        "--spike_firing_rate_weight", "0.0",
        "--temporal_readout", "learned",
        "--action_head_type", "ann",                       # ★ ACT 固定,全臂相同
        "--no_require_feature_enhancer_preload",
        "--no_load_text_projection_from_init",
        "--no_require_text_proj_preload",
        "--precision", "fp32",
        "--allow_hf_download" if False else "--no_allow_hf_download",
    ]

    # --- 文本冻结策略(全臂一致) ---
    argv += ["--freeze_text_encoder"] if text_mode == "frozen" else ["--train_text_encoder"]

    # --- KD(全臂开启;教师=官方 ckpt) ---
    if kd:
        argv += [
            "--teacher_checkpoint", str(OFFICIAL_CKPT),
            "--teacher_weight_source", "model",
            "--teacher_bert_path", str(BERT_PATH),
            "--distill_feature_weight", "0.25",
            "--distill_action_weight", "0.5",
        ]
    return argv


def module_classes(model) -> dict[str, str]:
    return {name: type(child).__name__ for name, child in model.named_children()}


def param_table(model) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for name, child in model.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        table[name] = {"total": total, "trainable": trainable}
    return table


def action_head_signature(model) -> str:
    """动作头的结构指纹:参数名+形状。全臂相同 ⇒ ACT 未变。"""
    head = dict(model.named_children()).get("action_head")
    if head is None:
        return "MISSING"
    payload = json.dumps(
        [(n, list(p.shape)) for n, p in head.named_parameters()], sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_arm(name: str, arm: dict[str, str], text_mode: str, kd: bool) -> dict:
    # 真实入口:turbovla.training.train_mixed(与 run_train_v2.sh 的 --module 一致)
    from turbovla.training.trainer import build_model_architecture
    from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats as parse_args

    argv = build_argv(arm, text_mode=text_mode, kd=kd)

    saved = sys.argv
    sys.argv = ["arm_build_check"] + argv
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    args.allow_hf_download = False
    # parse_args 无法表达"空字符串"(argparse 会当缺值),构建前手工压回 None
    if arm["vision"] == "dinov3":
        args.vision_pretrained_checkpoint = None
        args.vision_model_source_path = None

    model = build_model_architecture(args)
    classes = module_classes(model)
    table = param_table(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = {
        "arm": name,
        "combination": arm,
        "module_classes": classes,
        "params": table,
        "params_total": total,
        "params_trainable": trainable,
        "action_head_signature": action_head_signature(model),
        "config_echo": {
            "vision.encoder_type": args.vision_encoder_type,
            "text.encoder_type": args.text_encoder_type,
            "spike.downstream_type": args.downstream_type,
            "action_head_type": args.action_head_type,
        },
    }
    del model
    return report


def assert_imported_code() -> None:
    """确认解析到的是 v3 代码树,而不是环境里 editable 装的 v4 turbovla。"""
    import turbovla

    resolved = Path(turbovla.__file__).resolve()
    expected = os.environ.get("V3_CODE")
    print(f"[env] turbovla -> {resolved}")
    if expected:
        if not str(resolved).startswith(str(Path(expected).resolve())):
            raise SystemExit(
                f"[FATAL] 解析到错误的 turbovla: {resolved}(期望在 {expected} 下)。\n"
                f"3 请确认 PYTHONPATH 把 v3 代码树排在 editable 安装之前。"
            )
    elif "v2_code_bundle" not in str(resolved):
        raise SystemExit(
            f"[FATAL] 解析到错误的 turbovla: {resolved}\n"
            f"  请设 PYTHONPATH=<v3 code>:<v3 resources> 并(必要时)设 V3_CODE=<v3 code>。"
        )


def main() -> int:
    assert_imported_code()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS), help="逗号分隔,默认全部")
    ap.add_argument("--text-mode", default="frozen", choices=["frozen", "online"])
    ap.add_argument("--no-kd", action="store_true", help="关闭蒸馏(默认开启,与实验配方一致)")
    ap.add_argument("--check-teacher", action="store_true", help="额外验证 KD 教师能构建")
    ap.add_argument("--out", default="arm_build_report.json")
    args = ap.parse_args()

    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    reports, failures = [], []
    for name in names:
        if name not in ARMS:
            failures.append({"arm": name, "error": f"unknown arm (known: {list(ARMS)})"})
            continue
        print(f"\n===== {name}: {ARMS[name]} =====", flush=True)
        try:
            rep = run_arm(name, ARMS[name], args.text_mode, not args.no_kd)
        except Exception as exc:  # noqa: BLE001 - 逐臂隔离,失败不中断其余
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            failures.append({"arm": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        reports.append(rep)
        print(f"  模块类: {json.dumps(rep['module_classes'], ensure_ascii=False)}")
        print(
            f"  参数: 总 {rep['params_total']/1e6:.3f}M / 可训练 "
            f"{rep['params_trainable']/1e6:.3f}M | ACT 签名 {rep['action_head_signature']}"
        )
        for mod, cnt in sorted(rep["params"].items(), key=lambda kv: -kv[1]["total"]):
            print(f"    {mod:<28} {cnt['total']/1e6:>9.3f}M  (可训练 {cnt['trainable']/1e6:.3f}M)")

    if args.check_teacher:
        print("\n===== KD 教师(官方 ckpt) =====", flush=True)
        try:
            from turbovla.training.trainer import build_distillation_teacher
            from turbovla.training.train_mixed import (
                parse_args_with_mixed_suite_stats as parse_args,
            )

            argv = build_argv(ARMS["A3"], text_mode=args.text_mode, kd=True)
            saved, sys.argv = sys.argv, ["arm_build_check"] + argv
            try:
                t_args = parse_args()
            finally:
                sys.argv = saved
            t_args.allow_hf_download = False
            teacher = build_distillation_teacher(t_args)
            t_classes = module_classes(teacher)
            t_total = sum(p.numel() for p in teacher.parameters())
            print(f"  教师模块类: {json.dumps(t_classes, ensure_ascii=False)}")
            print(f"  教师参数: {t_total/1e6:.3f}M")
            reports.append(
                {
                    "arm": "TEACHER",
                    "module_classes": t_classes,
                    "params_total": t_total,
                    "params_trainable": sum(
                        p.numel() for p in teacher.parameters() if p.requires_grad
                    ),
                }
            )
            del teacher
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] teacher: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            failures.append({"arm": "TEACHER", "error": f"{type(exc).__name__}: {exc}"})

    # 一致性断言:主矩阵四臂的融合类与 ACT 签名必须一致
    ann_arms = [r for r in reports if r.get("combination", {}).get("downstream") == "ann"]
    if ann_arms:
        fusion = {r["module_classes"].get("vision_language_interaction") for r in ann_arms}
        acts = {r["action_head_signature"] for r in ann_arms}
        print(f"\n[CHECK] ANN 臂融合模块类 = {fusion}")
        print(f"[CHECK] ANN 臂 ACT 签名   = {acts}")
        if len(fusion) != 1:
            failures.append({"arm": "CONSISTENCY", "error": f"融合模块不一致: {fusion}"})
        if len(acts) != 1:
            failures.append({"arm": "CONSISTENCY", "error": f"ACT 签名不一致: {acts}"})

    payload = {"reports": reports, "failures": failures}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {args.out}")
    if failures:
        print(f"[RESULT] FAIL — {len(failures)} 项失败")
        return 1
    print(f"[RESULT] PASS — {len(reports)} 项全部构建成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
