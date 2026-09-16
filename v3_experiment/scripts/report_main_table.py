#!/usr/bin/env python
"""消融主表聚合器:把四维(性能/参数/能耗/内存+延迟)合成一张表。

★ 性能数字**一律从每个 suite 的明细 JSON 重算**,不读 `summary.json` ——
  实测发现 A4 的 ema 目录里 summary.json 只有 2 个 suite(89.00%),而明细四件齐全
  (92.20%);信任 summary.json 会直接得出错误结论。本脚本会显式报告这种不一致。

用法:
  python report_main_table.py                 # 用内置映射(与 eval_ablation.sh 输出目录一致)
  python report_main_table.py --md out.md --json out.json
  python report_main_table.py --only A0p,A4   # 只出部分臂
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BASE = Path("/data/260010028/dwh_vla")
PKG = BASE / "v2_code_bundle_20260906/package"
RES = BASE / "v2_code_bundle_20260906/resources"
V3EXP = BASE / "v3_experiment"

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# 臂 → (显示名, 评测结果目录, 口径)。目录里应有 <suite>.json 明细。
ARMS: dict[str, dict] = {
    "A0": {
        "label": "A0 官方锚点 (DINOv3-B+BERT+ANN)",
        "dir": V3EXP / "results/anchor_official",
        "weight_source": "model_state_dict",
        "external": True,
    },
    "A4": {
        "label": "A4 完整 v3 (SDT-V3+sootspike+spike 融合)",
        "dir": PKG / "output/eval/v3_sootspike_step_80000_ema_clean_8f1084e/results",
        "weight_source": "ema",
    },
    "A0p": {
        "label": "A0' DINOv3-B+BERT+ANN",
        "dir": PKG / "output/eval/ablation_A0p_step_80000_ema/results",
        "weight_source": "ema",
    },
    "A1": {
        "label": "A1 SDT-V3+BERT+ANN",
        "dir": PKG / "output/eval/ablation_A1_step_80000_ema/results",
        "weight_source": "ema",
    },
    "A2": {
        "label": "A2 DINOv3-B+sootspike+ANN",
        "dir": PKG / "output/eval/ablation_A2_step_80000_ema/results",
        "weight_source": "ema",
    },
    "A3": {
        "label": "A3 SDT-V3+sootspike+ANN",
        "dir": PKG / "output/eval/ablation_A3_step_80000_ema/results",
        "weight_source": "ema",
    },
    "A4-raw": {
        "label": "A4 (raw 口径,补充)",
        "dir": PKG / "output/eval/v3_sootspike_step_80000_raw_clean_8f1084e/results",
        "weight_source": "raw",
        "supplementary": True,
    },
}

ORDER = ["A0", "A0p", "A1", "A2", "A3", "A4", "A4-raw"]


def ci95(successes: int, episodes: int) -> float:
    """正态近似的 95% 置信半宽(pp)。"""
    if episodes == 0:
        return float("nan")
    p = successes / episodes
    return 1.96 * math.sqrt(p * (1 - p) / episodes) * 100


def read_perf(results_dir: Path) -> dict | None:
    """从 suite 明细重算;同时报告 summary.json 是否与明细矛盾。"""
    if not results_dir.is_dir():
        return None
    per_suite: dict[str, dict] = {}
    for suite in SUITES:
        f = results_dir / f"{suite}.json"
        if not f.is_file():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        per_suite[suite] = {
            "successes": int(j["total_successes"]),
            "episodes": int(j["total_episodes"]),
        }
    if not per_suite:
        return None
    succ = sum(v["successes"] for v in per_suite.values())
    eps = sum(v["episodes"] for v in per_suite.values())
    out = {
        "per_suite": per_suite,
        "successes": succ,
        "episodes": eps,
        "success_rate": succ / eps if eps else float("nan"),
        "ci95_pp": ci95(succ, eps),
        "complete": len(per_suite) == len(SUITES),
    }
    # 与 summary.json 对账(只报警告,不采信)
    sf = results_dir / "summary.json"
    if sf.is_file():
        s = json.loads(sf.read_text(encoding="utf-8"))
        s_succ, s_eps = s.get("total_successes"), s.get("total_episodes")
        if (s_succ, s_eps) != (succ, eps):
            out["summary_mismatch"] = {
                "summary_json": f"{s_succ}/{s_eps}",
                "recomputed": f"{succ}/{eps}",
                "note": "以明细重算为准(summary.json 可能只聚合了部分 suite)",
            }
    return out


def read_params() -> dict[str, dict]:
    f = V3EXP / "arm_build_report.json"
    if not f.is_file():
        return {}
    rep = json.loads(f.read_text(encoding="utf-8"))
    return {r["arm"]: r for r in rep.get("reports", [])}


def read_measures() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ("measure_A4.json", "measure_arms_random_init.json"):
        f = V3EXP / name
        if f.is_file():
            for r in json.loads(f.read_text(encoding="utf-8")).get("reports", []):
                out[r["arm"]] = r
    return out


def fmt(v, spec=".2f", dash="—"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    return format(v, spec)


def build_table(only: list[str] | None = None) -> tuple[str, dict]:
    params = read_params()
    meas = read_measures()
    arms = [a for a in ORDER if (only is None or a in only)]
    if only:
        arms = [a for a in ORDER if a in only] + [a for a in only if a not in ORDER]

    rows, payload = [], {}
    lines = []
    lines.append("| 臂 | 性能 (成功/局) | 95% CI | 参数量 | MACs | ACs | 能量 | 增益 | 显存@1 | 延迟@1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        cfg = ARMS.get(a)
        if cfg is None:
            continue
        perf = read_perf(cfg["dir"])
        key = a.replace("-raw", "")
        # 官方锚点(A0)没有构建报告,但它与 KD 教师同架构 → 复用教师的参数统计
        p = params.get(key) or params.get(a) or (params.get("TEACHER") if cfg.get("external") else None)
        m = meas.get(key) or meas.get(a)
        b1 = (m or {}).get("batches", {}).get("1")
        lat = (m or {}).get("latency", {}).get("median_ms")

        perf_cell = "待评测"
        ci_cell = "—"
        if perf:
            perf_cell = f"{perf['successes']}/{perf['episodes']} = **{100*perf['success_rate']:.1f}%**"
            if not perf["complete"]:
                perf_cell += f" ⚠️仅{len(perf['per_suite'])}/4套件"
            ci_cell = f"±{perf['ci95_pp']:.1f}pp"
        row = {
            "arm": a,
            "label": cfg["label"],
            "weight_source": cfg["weight_source"],
            "perf": perf,
            "params_total_M": (p or {}).get("params_total", 0) / 1e6 if p else None,
            "params_trainable_M": (p or {}).get("params_trainable", 0) / 1e6 if p else None,
            "macs_G": b1["macs"] / 1e9 if b1 else None,
            "acs_G": b1["acs"] / 1e9 if b1 else None,
            "energy_mJ": b1["energy_mJ_weighted"] if b1 else None,
            "energy_dense_equiv_mJ": b1["energy_mJ_dense_equiv"] if b1 else None,
            "energy_gain_x": b1["energy_gain_x"] if b1 else None,
            "peak_MiB_b1": b1["peak_alloc_MiB"] if b1 else None,
            "latency_ms": lat,
            "action_head_sig": (p or {}).get("action_head_signature"),
            "notes": ["参数取自 KD 教师(=官方架构)"] if cfg.get("external") and p else [],
        }
        if perf and perf.get("summary_mismatch"):
            row["notes"].append("summary.json 与明细不一致:" + str(perf["summary_mismatch"]))
        rows.append(row)
        lines.append(
            f"| {a} | {perf_cell} | {ci_cell} | {fmt(row['params_total_M'],'.1f')}M | "
            f"{fmt(row['macs_G'])}G | {fmt(row['acs_G'])}G | {fmt(row['energy_mJ'])}mJ | "
            f"{fmt(row['energy_gain_x'])}× | {fmt(row['peak_MiB_b1'],'.0f')}MiB | {fmt(row['latency_ms'],'.1f')}ms |"
        )
    payload["rows"] = rows

    # 口径与一致性检查
    checks = []
    sigs = {r["arm"]: r["action_head_sig"] for r in rows if r["action_head_sig"]}
    if sigs:
        uniq = set(sigs.values())
        checks.append(("ACT 结构签名一致(" + ("✅" if len(uniq) == 1 else "❌") + ")",
                       f"{len(uniq)} 种: {sorted(uniq)}"))
    for r in rows:
        for n in r["notes"]:
            checks.append((f"{r['arm']} ⚠️", n))
    payload["checks"] = checks

    text = "\n".join(lines)
    if checks:
        text += "\n\n**一致性检查**\n" + "\n".join(f"- {k}: {v}" for k, v in checks)
    return text, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="逗号分隔的臂名")
    ap.add_argument("--md", default=str(V3EXP / "MAIN_TABLE_CN.md"))
    ap.add_argument("--json", default=str(V3EXP / "main_table.json"))
    args = ap.parse_args()

    table, payload = build_table([s.strip() for s in args.only.split(",")] if args.only else None)
    header = ("# 消融主表(四维)\n\n"
              "> 由 `scripts/report_main_table.py` 生成;性能从各 suite 明细重算(不读 summary.json)。\n"
              "> 口径:EMA-80k 为主(A0 官方锚点为 raw,外部参考);LIBERO 8f1084e · fp32 · seed 7 · 50 trials。\n\n")
    Path(args.md).write_text(header + table + "\n", encoding="utf-8")
    Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table)
    print(f"\n写出: {args.md}\n      {args.json}")
    missing = [r["arm"] for r in payload["rows"] if r["perf"] is None]
    if missing:
        print(f"[INFO] 尚未评测(待训练完成): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
