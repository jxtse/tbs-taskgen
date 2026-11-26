"""Daily report (Markdown) for a run: item funnel, drop/park reasons, model spend by model x stage,
breakers, calibration queue.  Written to <run>/reports/daily-<day>.md."""
from __future__ import annotations

import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from .calib import manual_command
from .budget import settled_call_usd, today
from .state import read_json


def _read_review_record(path, default):
    try:
        return read_json(path, default)
    except (UnicodeError, OSError):
        return default


def round_verdict(directory: Path) -> tuple[dict, list] | None:
    """Read the frozen verdict and attribute it only from evidence for this job and package."""
    record = _read_review_record(directory / "verdict.json", {}) or {}
    if not isinstance(record, dict):
        return None
    # The routed copy can reflect re-normalization under the job's contract.
    routed = _read_review_record(directory.parent / "verdict.json", {}) or {}
    if (isinstance(routed, dict) and directory.name == f"r{routed.get('round')}"
            and (not record or all(routed.get(k) == record.get(k) for k in ("job", "pkg_sha", "calib_epoch")))):
        record = routed
    verdict = record.get("_normalized", record)
    if not isinstance(verdict, dict) or verdict.get("verdict") not in ("pass", "revise", "drop", "human"):
        return None
    runs = _read_review_record(directory.parent / "runs.json", []) or []
    epoch, sha = record.get("calib_epoch"), record.get("pkg_sha")
    job_name = record.get("job")
    candidates = [r for r in runs if isinstance(r, dict)
                  and (r.get("job_name") == job_name if job_name else directory.name == f"r{r.get('n')}")]
    models = set()
    for job in candidates:
        if not sha or not job.get("pkg_sha") or epoch is None or job.get("calib_epoch") is None:
            continue  # incomplete historical bindings cannot supply a reviewer identity
        if (job.get("outcome") != "done" or job["pkg_sha"] != sha or epoch != job["calib_epoch"]
                or (job.get("n") is not None and directory.name != f"r{job['n']}")):
            if job_name:
                return None
            continue
        models.add(job.get("model"))
    if record.get("model"):
        models.add(record["model"])
    model = next(iter(models)) if len(models) == 1 else None
    if model and "opus" not in str(model).lower():
        return None
    issues = [{k: issue.get(k) for k in ("id", "category", "title", "evidence")}
              for issue in verdict.get("issues") or []
              if isinstance(issue, dict) and issue.get("severity") == "blocking"
              and issue.get("category") in ("a", "b", "a_consistency", "b_missing_conditions")]
    return dict(verdict=verdict["verdict"], epoch=record.get("calib_epoch"),
                pkg_sha=record.get("pkg_sha"), model=model,
                model_source="recorded" if model else "unverified"), issues


def prereview_summary(run: Path) -> dict:
    """Count distinct items per mode. Compare records only with the verdict of their own epoch and round."""
    path = run / "ledger.sqlite"
    if path.is_file():
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            return _prereview_summary(run, db)
    return _prereview_summary(run, None)


def _prereview_summary(run, db):
    rows = {mode: dict(items_checked=0, checks=0, findings_raw=0, findings_validated=0,
                       revisions_applied=0, prereview_usd=0., revision_usd=0., usd=0.,
                       errors=0, opus_verdicts=0, opus_blocking_ab=0, unverified_verdicts=0)
            for mode in ("off", "shadow", "enforce")}
    seen = defaultdict(set)
    for path in sorted((run / "items").glob("*/prereview/e*-r*.json")):
        record = _read_review_record(path, {}) or {}
        mode = record.get("mode") if isinstance(record, dict) else None
        if mode not in ("shadow", "enforce"):
            continue
        item = path.parents[1]
        seen[mode].add(item.name)
        row = rows[mode]
        row["checks"] += 1
        row["findings_raw"] += len(record.get("raw") or [])
        row["findings_validated"] += len(record.get("validated") or [])
        row["revisions_applied"] += bool(record.get("revision_applied"))
        for stage, call_ids, saved_field, column in (
            ("S9P", record.get("call_ids") or record.get("call_id"), "usd", "prereview_usd"),
            ("S8", record.get("revision_call_id"), "revision_usd", "revision_usd")
        ):
            usd = settled_call_usd(db, call_ids, item.name, stage) if db and call_ids else None
            row[column] += (record.get(saved_field) or 0.) if usd is None else usd
        row["errors"] += bool(record.get("error"))
        result = round_verdict(item / "review" / f"r{record.get('round')}")
        if not result or result[0]["epoch"] not in (None, record.get("epoch")):
            continue
        if result[0]["model_source"] == "unverified" or result[0]["epoch"] is None:
            row["unverified_verdicts"] += 1
        else:
            row["opus_verdicts"] += 1
            row["opus_blocking_ab"] += len(result[1])
    for mode, row in rows.items():
        row["items_checked"] = len(seen[mode])
        row["usd"] = round(row["prereview_usd"] + row["revision_usd"], 8)
        for component in ("prereview_usd", "revision_usd"):
            row[component] = round(row[component], 8)
    return rows


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
