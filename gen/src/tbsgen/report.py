"""Daily report (Markdown) for a run: item funnel, drop/park reasons, model spend by model x stage,
breakers, calibration queue.  Written to <run>/reports/daily-<day>.md."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .budget import today
from .state import read_json


def daily_report(P, day: str | None = None) -> Path:
    day = day or today()
    st = P.status()
    led = P.ledger.summary(day)
    led_all = P.ledger.summary(None)
    ev = [e for e in P.store.events()]
    stages_done = Counter(e["stage"] for e in ev if e["kind"] == "stage_end")
    lines = [f"# tbsgen 日报 {day}", "", f"运行目录：`{P.store.root}`", "",
             "## 条目状态", "", "| 状态 | 数 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(st["by_status"].items())]
    lines += ["", "## 当前阶段 × 状态", "", "| 阶段/状态 | 数 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(st["by_stage"].items())]
    lines += ["", "## 丢弃 / 暂停 / 暂存原因", "", "| 原因 | 数 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(st["reasons"].items(), key=lambda kv: -kv[1])] or ["| – | 0 |"]
    lines += ["", "## 阶段完成次数（累计）", "", "| 阶段 | 次数 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(stages_done.items())]
    lines += ["", f"## 模型花费（{day}，北京时间日）", "",
              "| 模型 | 阶段 | 真实 | 调用 | 失败 | 输入 tok | 输出 tok | 缓存读 | 美元 |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['stage']} | {'是' if r['real'] else '否'} | {r['calls']} | {r['errors']} | "
              f"{r['in_tok']} | {r['out_tok']} | {r['cache_read']} | {r['usd']:.4f} |" for r in led["rows"]]
    b = P.cfg["budget"]
    lines += ["", f"今日合计：{led['calls_total']} 次，${led['usd_total']:.4f}；累计：{led_all['calls_total']} 次，"
                  f"${led_all['usd_total']:.4f}。日预算：{b['daily_usd_total'] if b['daily_usd_total'] is not None else '未设置（真实模型不可调用）'}。",
              "价格为占位值（DESIGN §3.1），以实际账单为准。"]
    lines += ["", "## 熔断"]
    lines += [f"- {x['model']}: {x['reason']}" for x in led["breakers_open"]] or ["- 无"]
    waiting = [i for i in P.store.items() if i.state["status"] == "waiting"]
    lines += ["", "## 校准队列（S9，未自动提交）"]
    for i in waiting:
        runs = read_json(i.p("calib", "runs.json"), [])
        if runs:
            lines.append(f"- {i.item_id}: {runs[-1]['who']} 第 {runs[-1]['n']} 次 → `{runs[-1]['command']}`")
    if not waiting:
        lines.append("- 无")
    out = P.store.root / "reports" / f"daily-{day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
