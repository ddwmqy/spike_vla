#!/usr/bin/env python
"""端到端校验:把 `run_train_v2_ablation.sh` DRY_RUN 输出的真实命令行喂回 trainer。

校验链:shell 拼串 → mixed-suite 参数解析 → 模型构建。任何"多余参数/参数错位/
轴没生效"都会在这里暴露,而不是在 13 小时的训练里。

用法:
  PYTHONPATH=<code>:<resources> python check_arm_cli.py /tmp/dryrun_A0p.txt [...]
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path


def argv_from_dryrun(path: Path) -> list[str]:
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    tokens = shlex.split(line)
    if "--module" not in tokens:
        raise SystemExit(f"{path}: 未找到 --module,不是 dry-run 输出?")
    idx = tokens.index("--module")
    return tokens[idx + 2 :]  # 跳过 turbovla.training.train_mixed


def check(path: Path) -> dict:
    from turbovla.models.turbovla import build_turbovla
    from turbovla.training.trainer import build_model_architecture
    from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats as parse_args

    argv = argv_from_dryrun(path)
    empties = [i for i, a in enumerate(argv) if a == ""]
    saved, sys.argv = sys.argv, ["check_arm_cli"] + argv
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    args.allow_hf_download = False

    model = build_model_architecture(args)
    classes = {n: type(c).__name__ for n, c in model.named_children()}
    total = sum(p.numel() for p in model.parameters())
    result = {
        "dry_run_file": str(path),
        "empty_args_at": empties,
        "resolved": {
            "vision": args.vision_encoder_type,
            "vision_model_path": args.vision_model_path,
            "vision_pretrained_checkpoint": repr(args.vision_pretrained_checkpoint),
            "vision_model_source_path": repr(args.vision_model_source_path),
            "text": args.text_encoder_type,
            "bert_path": args.bert_path,
            "text_source_path": repr(args.text_source_path),
            "downstream": args.downstream_type,
            "action_head": args.action_head_type,
            "image_size": args.vision_image_size,
            "freeze_backbones": args.freeze_backbones,
            "freeze_text_encoder": args.freeze_text_encoder,
            "distill": (args.distill_feature_weight, args.distill_action_weight),
            "teacher": Path(args.teacher_checkpoint).name,
            "max_steps": args.max_steps,
        },
        "classes": classes,
        "params_total": total,
        "wrong_package": "v2_code_bundle" not in str(build_turbovla.__module__),
    }
    del model
    return result


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out, bad = [], 0
    for p in sys.argv[1:]:
        print(f"\n===== {p} =====", flush=True)
        try:
            rep = check(Path(p))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            out.append({p: f"{type(exc).__name__}: {exc}"})
            bad += 1
            continue
        out.append(rep)
        for k, v in rep["resolved"].items():
            print(f"  {k:<28} {v}")
        print(f"  模块类 {json.dumps(rep['classes'], ensure_ascii=False)}")
        print(f"  总参数 {rep['params_total']/1e6:.3f}M | 空参数位置 {rep['empty_args_at'] or '无'}")
    Path("/tmp/arm_cli_check.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[RESULT] {'FAIL' if bad else 'PASS'} — {len(out) - bad}/{len(out)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
