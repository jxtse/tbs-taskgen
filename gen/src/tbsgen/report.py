"""Daily report (Markdown) for a run: item funnel, drop/park reasons, model spend by model x stage,
breakers, calibration queue.  Written to <run>/reports/daily-<day>.md."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

from .calib import manual_command
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
    lines += ["", f"待人工核实的远端作业（remote_unknown）：{st['remote_unknown']}"]
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
              "价格为占位值（config.py models.prices），以实际账单为准。"]
    lines += ["", "## 熔断"]
    lines += [f"- {x['model']}: {x['reason']}" for x in led["breakers_open"]] or ["- 无"]
    lines += ["", "## 待人工提交的校准与审核作业"]
    commands = []
    for it in P.store.items():
        stage = it.state["stage"]
        if stage not in ("S9", "S9R") or it.state["status"] != "waiting":
            continue
        runs = read_json(it.p("review" if stage == "S9R" else "calib", "runs.json"), [])
        if runs and runs[-1].get("state") in ("intent", "reserved") and not runs[-1].get("launch_attempted"):
            rec = runs[-1]
            attempt = rec.get("attempt", rec["n"])
            command = manual_command(it.dir, rec["job_name"], attempt, rec.get("config_path") or P.cfg.get("config_path"))
            commands.append(f"- {it.item_id} · {rec['who']} · 第 {attempt} 次 → `{command}`")
    lines.extend(commands or ["- 无"])
    out = P.store.root / "reports" / f"daily-{day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out

def throughput(P, days=None, *, now=None):
    """Integrate observed item states over the selected interval (mean jobs = job-seconds / seconds)."""
    now = time.time() if now is None else now
    items = P.store.items()
    histories = [sorted((h for h in it.state.get("history", []) if h.get("ts") is not None),
                        key=lambda h: h["ts"]) for it in items]
    if days is not None and days <= 0:
        raise ValueError("days must be positive")
    first = min([now] + [it.state.get("created", now) for it in items] +
                [h[0]["ts"] for h in histories if h])
    start = now - days * 86400 if days else first
    waits = defaultdict(Counter)
    jobs = Counter(dict.fromkeys(("S4c", "S5G", "S9", "S9R"), 0.0))
    released, funnel = Counter(), Counter()
    for it, history in zip(items, histories):
        seen = set()
        for idx, h in enumerate(history):
            ts, stage = h["ts"], h.get("stage")
            if ts > now or not stage:
                continue
            last = idx + 1 == len(history)
            end = now if last else history[idx + 1]["ts"]
            if last and h.get("result") in ("waiting", "paused", "parked") and it.state["status"] != h["result"]:
                end = min(end, it.state.get("updated", end))
            length = max(0, min(now, end) - max(ts, start))
            if ts >= start or (length and h.get("result") in ("waiting", "paused", "parked")):
                seen.add(stage)
            if stage == "S10" and h.get("result") == "ok" and ts >= start:
                released[today(ts)] += 1
            if not length or h.get("result") not in ("waiting", "paused", "parked"):
                continue
            reason = h.get("reason") or "unknown"
            if reason.endswith(":agent_running") or (stage in ("S9", "S9R") and
                                                     reason.startswith(("submitted ", "awaiting "))):
                jobs[stage] += length
                continue
            if "agent slots full" in reason:
                reason = "queued: agent slots full"
            elif "slots full" in reason:
                reason = "queued: slots full"
            elif reason.startswith("relay_down"):
                reason = "relay_down"
            elif reason.startswith("budget:"):
                reason = reason.split()[0]
            waits[stage][reason] += length / 3600
        if it.state.get("created", now) <= now and (it.state.get("updated", now) >= start or
                                                    it.state["status"] in ("active", "waiting", "paused", "parked")):
            seen.add(it.state["stage"])
        funnel.update(seen)
    duration = now - start
    return dict(since=start, until=now, waiting_item_hours={s: dict(v) for s, v in waits.items()},
                mean_concurrent_jobs={s: n / duration if duration else 0 for s, n in jobs.items()},
                releases_per_day=dict(released), funnel=dict(funnel))
