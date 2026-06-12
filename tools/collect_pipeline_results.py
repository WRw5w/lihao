"""Aggregate exp_pipelines/*/lora/history.json into exp_pipelines/RESULTS.md.

Run from the project root: python tools/collect_pipeline_results.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp_pipelines"
BASELINE_HISTORY = ROOT / "outputs" / "lora" / "history.json"


def best_row(history: list[dict]) -> dict | None:
    rows = [h for h in history if "mid_03_06" in h]
    return max(rows, key=lambda h: h["mid_03_06"]) if rows else None


def fmt(row: dict | None, epochs_seen: int) -> str:
    if row is None:
        return f"| (无验证记录, {epochs_seen} 轮) | - | - | - | - |"
    return (f"| {row['epoch']}/{epochs_seen} | {row.get('mid_03_06', '-')} "
            f"| {row.get('noisy_all', '-')} | {row.get('low_lt03', '-')} | {row.get('high_ge06', '-')} |")


def main() -> None:
    lines = [
        "# 流程方案对比结果",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| 方案 | best 轮次 | mid_03_06（主指标） | noisy_all | low_lt03 | high_ge06 |",
        "|---|---|---|---|---|---|",
    ]
    if BASELINE_HISTORY.exists():
        hist = json.loads(BASELINE_HISTORY.read_text(encoding="utf-8"))
        rows = [h for h in hist if "mid_03_06" in h]
        if rows:
            upto12 = max((h for h in rows if h["epoch"] <= 12), key=lambda h: h["mid_03_06"])
            full = max(rows, key=lambda h: h["mid_03_06"])
            lines.append("| A 基准（≤12 轮） " + fmt(upto12, 12)[1:])
            lines.append("| A 基准（全程） " + fmt(full, max(h['epoch'] for h in rows))[1:])
    for d in sorted(p for p in EXP.iterdir() if p.is_dir()):
        hist_path = d / "lora" / "history.json"
        if not hist_path.exists():
            lines.append(f"| {d.name} | 未完成 | - | - | - | - |")
            continue
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        epochs_seen = max((h.get("epoch", 0) for h in hist), default=0)
        lines.append(f"| {d.name} " + fmt(best_row(hist), epochs_seen)[1:])
    lines += [
        "",
        "判读：mid_03_06 最高者胜出；low_lt03 显著上升 = 拟合噪声预警。",
        "方案设计与协议见 README.md；各方案动机见各目录 PLAN.md。",
    ]
    out = EXP / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
