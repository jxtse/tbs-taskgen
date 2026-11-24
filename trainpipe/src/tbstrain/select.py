from collections import Counter, defaultdict
from math import floor

from .core import digest


def mix(pools, shares):
    """Largest feasible cohort with largest-remainder integer tier quotas."""
    weights = {k: v for k, v in shares.items() if v > 0}
    total = sum(weights.values())
    for n in range(sum(len(pools[k]) for k in weights), 0, -1):
        raw = {k: n * v / total for k, v in weights.items()}
        quotas = {k: floor(x) for k, x in raw.items()}
        for k in sorted(raw, key=lambda k: (quotas[k] - raw[k], k))[: n - sum(quotas.values())]:
            quotas[k] += 1
        if all(len(pools[k]) >= q for k, q in quotas.items()):
            return [r for k, q in quotas.items() for r in pools[k][:q]]
    return []


def easy_capped(pools, easy_max):
    """All medium and hard trajectories, plus easy ones up to `easy_max` of the final mix (best quality first)."""
    core = pools.get("hard", []) + pools.get("medium", [])
    if easy_max >= 1:
        return core + pools.get("easy", [])
    n_easy = floor(easy_max * len(core) / (1 - easy_max)) if core else 0
    easy = sorted(pools.get("easy", []), key=lambda r: (-r["quality"], r["rank"]))
    return core + easy[:n_easy]


def select(ctx):
    cfg = ctx.cfg.get("select", {})

    def build():
        tasks = {t["task_id"]: t for t in ctx.rows("tasks")}
        tiers = {t["task_id"]: t["tier"] for t in ctx.rows("difficulty")}
        tasks_judged = {r["task_id"]: r["verdict"] for r in ctx.rows("judge-task")}
        judged = {r["trial_id"]: r for r in ctx.rows("judge-trajectory")}
        shares = cfg.get("shares", {"hard": 0.45, "medium": 0.4, "easy": 0.15})
        mode = cfg.get("mode", "easy_cap")       # easy_cap: all medium/hard + easy up to easy_max; shares: exact mix
        easy_max = cfg.get("easy_max", 0.15)
        if mode not in ("easy_cap", "shares") or not 0 <= easy_max <= 1:
            raise ValueError("invalid selection mode")
        cap = cfg.get("per_task", 2)
        max_share = cfg.get("max_token_share", 0.02)
        if (
            not shares
            or set(shares) - {"easy", "medium", "hard"}
            or min(shares.values()) < 0
            or not sum(shares.values())
            or not 0 < max_share <= 1
            or cap < 1
        ):
            raise ValueError("invalid selection constraints")
        reasons = {}
        by_task = defaultdict(list)
        rows = list(ctx.rows("screen"))
        for r in rows:
            tid, uid = r["task_id"], r["trial_id"]
            verdict = judged.get(uid, {})
            why = None
            if r["disposition"] == "drop":
                why = "screen: " + ", ".join(r["reasons"])
            elif not r["within_limit"]:
                why = "max_length"
            elif tasks_judged.get(tid) != "keep":
                why = "task_judge"
            elif verdict.get("verdict") != "keep":
                why = "trajectory_judge"
            elif not shares.get(tiers[tid]):
                why = "tier"
            elif tiers[tid] == "easy" and verdict["quality"] < cfg.get("easy_quality", 4):
                why = "easy_quality"
            if why:
                reasons[uid] = why
                continue
            by_task[tid].append(
                {**r, "tier": tiers[tid], "domain": tasks[tid]["domain"], "quality": verdict["quality"]}
            )
        pools = defaultdict(list)
        seed = cfg.get("seed", 0)

        def tie(r):
            return digest([seed, r["trial_id"]])

        for tid, candidates in by_task.items():
            used = set()
            for rank in range(min(cap, len(candidates))):
                best = min(candidates, key=lambda r: (r["harness"] in used, r["supervised_tokens"], tie(r)))
                candidates.remove(best)
                used.add(best["harness"])
                pools[best["tier"]].append({**best, "rank": rank})
            for r in candidates:
                reasons[r["trial_id"]] = "per_task_cap"
        for pool in pools.values():
            pool.sort(key=lambda r: (r["rank"], tie(r)))
        def draw():
            return easy_capped(pools, easy_max) if mode == "easy_cap" else mix(pools, shares)

        while selected := draw():
            sums = Counter()
            for r in selected:
                sums[r["task_id"]] += r["supervised_tokens"]
            # n tasks cannot all stay under 1/n of the tokens: `token_share_floor` (x the even share, 0 = off)
            # keeps a small pilot set from emptying; production sets with hundreds of tasks never hit it.
            effective = max(max_share, cfg.get("token_share_floor", 0) / len(sums))
            maximum = sum(sums.values()) * effective
            offenders = {tid for tid, n in sums.items() if n > maximum + 1e-9}
            if not offenders:
                break
            worst = max(
                (r for r in selected if r["task_id"] in offenders),
                key=lambda r: (sums[r["task_id"]], r["rank"], r["supervised_tokens"], tie(r)),
            )
            pools[worst["tier"]].remove(worst)
            reasons[worst["trial_id"]] = "token_share_cap"
        chosen = {r["trial_id"] for r in selected}
        ledger = [
            dict(
                trial_id=r["trial_id"],
                task_id=r["task_id"],
                keep=r["trial_id"] in chosen,
                reason="selected" if r["trial_id"] in chosen else reasons.get(r["trial_id"], "tier_quota"),
            )
            for r in rows
        ]
        summary = {k: dict(Counter(r[k] for r in selected)) for k in ("tier", "domain", "harness")}
        summary["tasks"] = len({r["task_id"] for r in selected})
        summary["supervised_tokens"] = sum(r["supervised_tokens"] for r in selected)
        ctx.emit("selection", selected)
        ctx.emit("select.ledger", ledger)
        report = [
            "# Selection",
            "",
            f"Selected {len(chosen)} trajectories from {summary['tasks']} tasks, "
            f"{summary['supervised_tokens']} supervised tokens (mode {mode}).",
            "",
            "## Exclusion counts",
            "",
            *[f"- {k} — {n}" for k, n in Counter(r["reason"] for r in ledger if not r["keep"]).most_common()],
        ]
        for k in ("tier", "domain", "harness"):
            report += ["", "## " + k, ""] + [
                f"- {v or '(unknown)'} — {n}" for v, n in sorted(summary[k].items())
            ]
        report += ["", "## Exclusions", ""] + [
            f"- {r['trial_id']} — {r['reason']}" for r in ledger if not r["keep"]
        ]
        (ctx.work / "selection_report.md").write_text("\n".join(report) + "\n", "utf-8")
        return {"count": len(selected), **summary}

    parents = [ctx.dependency(k) for k in ("difficulty", "screen", "judge-task", "judge-trajectory")]
    return ctx.stage("select", [parents, cfg], build)
