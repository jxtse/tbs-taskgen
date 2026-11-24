from collections import Counter as Frequencies
from pathlib import Path
import posixpath
import re
import shlex

from .core import atomic, digest, file_hash, lines
from .profile import ProfileError, configured_profiles
from .native import CONVERTERS, message, segment
from .tasks import valid
from .wire import Counter, check, export_modes, image_dependents, render, string_content

HIDDEN = re.compile(
    # absolute grader paths only: `/tests/...` as a path, not `d + '/test'` (a relative directory being built)
    r"(?:^|[\s'\"=(:,])/(?:tests|solution|oracle|logs/verifier|installed-agent)(?=/|[\s'\")]|$)"
    r"|\bverifier (?:output|reward|result)\s*[:=]",
    re.I | re.M,
)
SEARCH = re.compile(
    r"\b(?:find|grep|rg|locate|ls)\b[^;&|\n]*?(?:grader|truth|answer|solution|tests?\b|verifier|oracle|expected|hidden|reward|label|reference|benchmark)",
    re.I,
)
OUTSIDE = re.compile(r"(?:^|[\s'\"=])/(?:$|[\s'\"*]|(?:root|home|opt|srv|mnt|logs)(?:/|\b))|(?:^|/)\.\./")
WHOLE = re.compile(r"\b(?:find|(?:grep|rg)\s+(?:-[\w]*[rR][\w]*\s+)?)\s*[^;&|\n]*?\s/(?:\s|$)")
NET = re.compile(r"\b(?:curl|wget|(?:pip\d*|pip\s+|uv\s+pip)\s+install)\b", re.I)
FAIL = re.compile(
    r"^(?:error[: ]*)?(?:no (?:matches|results|files)(?: found)?|permission denied|not found|no such file or directory|command not found|network (?:disabled|unreachable))\.?$",
    re.I,
)


SENSITIVE = re.compile(
    r"(?:grad(?:er|ing)|truth|answer|solution|tests?\b|verifier|oracle|hidden|reward|reference)", re.I
)
PATH_TOKEN = re.compile(r"""(?:^|[\s"'=])(/[^\s"';|&]*)""")
READ_OP = re.compile(r"\b(?:find|ls|head|cat|tail|grep|glob|rg|read|locate|open)\b", re.I)
# Searches for grader material in places the agent did not create. Validated on real Claude Code calibration
# trajectories: the agent's own scratch files (/tmp/*/truth.pkl, /app/work/score.py) are not probes.
PROBE = re.compile(
    r"\b(?:find|locate|ls|grep\s+-\w*r\w*|rg)\b[^|;&\n]*?(?:grad(?:er|ing)|truth|answer|solution|expected|oracle|"
    r"tests?\b|verifier|reward|hidden|secret|label|benchmark|reference)",
    re.I,
)
PROBE_TARGET = re.compile(
    r"\b(?:find|ls(?:\s+-\w+)*|grep\s+-\w*r\w*[^|;&\n]*|rg[^|;&\n]*)\s+/(?:\s|$|logs\b|tests\b|solution\b|oracle\b|"
    r"opt\b|root\b|home\b|srv\b|mnt\b|installed-agent\b)"
)
SCRATCH = ("/app", "/work", "/tmp", "/root/.cache", "/logs/agent")
SEARCH_CMDS = {"find", "locate", "ls", "grep", "egrep", "rg", "tree", "du", "cat", "head", "tail"}
GRADER_ROOTS = ("/logs/verifier", "/tests", "/solution", "/oracle", "/installed-agent")
SYSTEM_ROOTS = ("/opt", "/root", "/home", "/srv", "/mnt", "/logs")
SENSITIVE_NAME = re.compile(
    r"grad(?:er|ing)|truth|answer|solution|expected|oracle|verif|reward|hidden|secret|label|benchmark|reference|inject",
    re.I,
)


HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1\b", re.S)


def _subcommands(command):
    """Yield tokenized commands with their terminating shell operator."""
    command = HEREDOC.sub("", command)
    argv = []

    def flush(argv):
        while argv and (("=" in argv[0] and not argv[0].startswith("-")) or argv[0] in ("sudo", "time", "env", "nohup", "nice")):
            argv = argv[1:]
        if argv and argv[0] == "timeout":     # timeout [-s SIG] [-k D] DURATION cmd ...
            argv = argv[1:]
            while argv and argv[0].startswith("-"):
                argv = argv[2:] if argv[0] in ("-s", "-k", "--signal", "--kill-after") else argv[1:]
            argv = argv[1:]
        return argv

    tokens = []
    for line in command.split("\n"):
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            tokens.extend(list(lexer))
        except ValueError:          # unbalanced quote in this line: plain split
            tokens.extend(line.split())
        tokens.append(";")
    for tok in tokens:
        if tok == "\n" or (tok and set(tok) <= set(";&|()\n")):
            for op in re.findall(r"&&|\|\||\|&|[;&|()]", tok):
                yield flush(argv), op
                argv = []
        elif tok and set(tok) <= set("<>"):
            argv.append(tok)
        else:
            argv.append(tok)
    argv = flush(argv)
    if argv:
        yield argv, ";"


def _under(path, roots):
    return any(path == r or path.startswith(r + "/") for r in roots)


def _changed_directory(state, argv):
    base, stack = state
    args = [a for a in argv[1:] if a not in {"--", "-L", "-P"}]
    if argv[0] in ("cd", "pushd") and len(args) == 1 and not args[0].startswith(("-", "+")):
        if argv[0] == "pushd":
            stack = (*stack, base)
        return posixpath.normpath(posixpath.join(base, args[0])), stack
    if argv[0] == "popd" and not args and stack:
        return stack[-1], stack[:-1]
    return state


def _commands_with_cwd(command, base):
    states = {(posixpath.normpath(base), ())}
    scopes = []
    skipped_success = set()
    skipped_failure = set()
    previous_op = None
    for argv, op in _subcommands(command):
        piped = previous_op in ("|", "|&") or op in ("|", "|&", "&")
        previous_op = op
        if argv:
            yield argv, {base for base, _ in states}, bool(skipped_success)
        changed = states
        if argv and not piped and argv[0] in ("cd", "pushd", "popd"):
            changed = {_changed_directory(state, argv) for state in states}
        # A successful cd changes cwd. Its failure path starts at the old cwd.
        success = changed | skipped_success
        failure = states | skipped_failure
        skipped_success, skipped_failure = set(), set()
        if op == "||":
            states, skipped_success = failure, success
        elif op == "&&":
            states, skipped_failure = success, failure
        else:
            states = success
        if op == "(":
            scopes.append(states)
        elif op == ")" and scopes:
            states = scopes.pop()


def _probe_at(argv, base, task_dirs):
    cmd = argv[0].rsplit("/", 1)[-1]
    paths = [posixpath.normpath(posixpath.join(base or "", a)) for a in argv[1:]
             if not a.startswith("-") and (base is not None or posixpath.isabs(a))]
    # reading grader directories directly, by any command
    if any(_under(p.rstrip("/*"), GRADER_ROOTS) for p in paths) and cmd not in ("mkdir", "echo", "printf"):
        return True
    if cmd not in SEARCH_CMDS:
        return False
    recursive = cmd in ("find", "locate", "tree", "du") or any(
        a.startswith("-") and not a.startswith("--") and "r" in a.lower() for a in argv[1:]
    ) or cmd == "rg"
    foreign = [
        p for p in paths
        if not _under(p.rstrip("/"), (*task_dirs, *SCRATCH)) and (p == "/" or _under(p, SYSTEM_ROOTS))
    ]
    names = " ".join(argv[1:])
    if cmd == "locate" and SENSITIVE_NAME.search(names):
        return True
    if recursive and "/" in paths and (cmd == "find" or SENSITIVE_NAME.search(names)):
        return True     # whole-disk search
    if foreign and (recursive or cmd == "ls") and SENSITIVE_NAME.search(names):
        return True
    return False


def shell_probe(command, task_dirs, cwd=None):
    """Return None for a possible probe whose relative path depends on a branch."""
    if HIDDEN.search(command) and ("||" not in command or any(HIDDEN.search(m[0]) for m in HEREDOC.finditer(command))):
        return True
    result = False
    for argv, bases, conditional in _commands_with_cwd(command, posixpath.join(task_dirs[0], cwd or ".")):
        if conditional and argv[0] in ("cd", "pushd", "popd"):
            continue
        # Absolute and inline paths have the same meaning under every candidate cwd.
        if any(HIDDEN.search(arg) for arg in argv) or _probe_at(argv, None, task_dirs):
            return True
        if any(_probe_at(argv, base, task_dirs) for base in bases):
            if not conditional and len(bases) == 1:
                return True
            result = None
    return result


def shell_input(args, default_dir):
    """Read shell text or an argv vector from a native tool payload."""
    cwd = default_dir
    if isinstance(args, dict):
        cwd = args.get("cwd") or args.get("workdir") or default_dir
        if isinstance(args.get("action"), dict):
            return shell_input(args["action"], cwd)
        for key in ("command", "cmd", "script", "input"):
            if key in args:
                args = args[key]
                break
        else:
            options = {"cwd", "workdir", "description", "shell"}
            candidates = [v for k, v in args.items() if k not in options and isinstance(v, (str, list))]
            if len(candidates) != 1:
                raise ValueError("ambiguous shell command")
            args = candidates[0]
    if isinstance(args, list) and args:
        if len(args) >= 3 and args[0].rsplit("/", 1)[-1] in {"sh", "bash", "zsh", "dash"} and args[1] in ("-c", "-lc"):
            args = args[2]
        else:
            args = shlex.join(args)
    if not isinstance(args, str):
        raise ValueError("missing shell command")
    return args, cwd


def inspect_segment(source, offline, task_dirs=("/app", "/work")):
    results = {
        b["tool_use_id"]: b for m in source["messages"] for b in m["content"] if b["type"] == "tool_result"
    }
    flags = set()
    fatal = set()
    strip = set()
    for result in results.values():
        if HIDDEN.search(string_content(result["content"])):
            flags.add("hidden_access")
            fatal.add("hidden_access")
    for m in source["messages"]:
        for b in m["content"]:
            if b["type"] != "tool_use":
                continue
            name = b["name"].lower().rsplit(".", 1)[-1].rsplit("__", 1)[-1]
            args = b["input"]
            value = " ".join(str(v) for v in args.values()) if isinstance(args, dict) else str(args)
            shell = name in {"bash", "exec_command", "shell", "local_shell"}
            if shell:
                value, cwd = shell_input(args, task_dirs[0])
            result = results.get(b["id"])
            output = string_content(result["content"]).strip() if result else ""
            simple = not re.search(r"[;&|\n]", value)
            no_matches = not output and (result or {}).get("is_error", False)
            if not output and re.match(r"^\s*(find|grep|rg|locate)\b", value):
                no_matches = True
            immaterial = result is not None and simple and (no_matches or bool(FAIL.fullmatch(output)))
            if shell:
                probe = shell_probe(value, task_dirs, cwd)
                if probe is None:
                    flags.add("inconclusive")
                # only when curl/wget/pip install is the command itself, not a grep pattern mentioning it
                shell_net = any(
                    argv[0].rsplit("/", 1)[-1] in ("curl", "wget")
                    or (argv[0].rsplit("/", 1)[-1] in ("pip", "pip3", "uv", "conda", "mamba") and "install" in argv[1:3])
                    or (argv[0].startswith("python") and argv[1:4] == ["-m", "pip", "install"])
                    for argv, _ in _subcommands(value) if argv
                )
            else:
                probe = bool(HIDDEN.search(value) and READ_OP.search(name + " " + value))
                shell_net = bool(NET.search(value))
            network = offline and (name in ("websearch", "webfetch", "web_search", "web_fetch") or shell_net)
            # Read/Glob/Grep tools: a sensitive path outside the task and scratch areas.
            if name in ("read", "glob", "grep", "ls") and isinstance(args, dict):
                base = posixpath.join(task_dirs[0], args.get("cwd") or args.get("workdir") or ".")
                targets = [args[k] for k in ("file_path", "path", "pattern") if isinstance(args.get(k), str)]
                paths = [posixpath.normpath(posixpath.join(base, t)) for t in targets or ["."]]
                if any(_under(p, GRADER_ROOTS) for p in paths) or (
                    SENSITIVE.search(value) and any(not _under(p, (*task_dirs, *SCRATCH)) for p in paths)
                ):
                    probe = True
            if network:
                flags.add("network")
                # Empty successful downloads can write a file. Only explicit failures can be stripped.
                if not result or not (result.get("is_error") or FAIL.fullmatch(output)) or not immaterial:
                    fatal.add("network")
                else:
                    strip.add(b["id"])
            if probe:
                flags.add("answer_search")
                if immaterial:
                    strip.add(b["id"])
                else:
                    fatal.add("answer_search")
    if strip and not fatal:
        cleaned = []
        for m in source["messages"]:
            blocks = m["content"]
            if any(b.get("type") == "tool_use" and b["id"] in strip for b in blocks):
                # Drop accompanying narrative about the probe. Retain independent parallel calls.
                blocks = [b for b in blocks if b["type"] == "tool_use" and b["id"] not in strip]
            else:
                blocks = [b for b in blocks if b.get("tool_use_id") not in strip]
            if blocks:
                cleaned.append({**m, "content": blocks})
        source["messages"] = cleaned
    return flags, fatal, strip


def inspect_omitted_events(events, harness, offline, task_dirs):
    """Audit tools even when replay removal or tail truncation omits them from training."""
    blocks = []
    network = False
    for event in events:
        native = event.get("message", {})
        content = native.get("content", [])
        if harness == "claude-code" and isinstance(content, list):
            blocks.extend(b for b in content if b.get("type") in {"tool_result", "tool_use"})
        elif harness == "pi":
            native = event.get("event", {}).get("message", native)
            if native.get("role") == "toolResult":
                blocks.append(
                    dict(
                        type="tool_result",
                        tool_use_id=native["toolCallId"],
                        content=native["content"],
                        is_error=native.get("isError", False),
                    )
                )
            elif native.get("role") == "assistant":
                for b in native["content"]:
                    if b["type"] == "toolCall":
                        blocks.append(dict(type="tool_use", id=b["id"], name=b["name"], input=b["arguments"]))
        elif harness == "codex" and event.get("payload", {}).get("type") in {
            "web_search_call",
            "web_search_end",
        }:
            network = True
    flags, fatal, removed = inspect_segment(segment([message("user", blocks)]), offline, task_dirs)
    if offline and network:
        flags.add("network")
        fatal.add("network")
    return flags, fatal, removed


def screen(ctx):
    ctx.check_sources("trials")
    cfg = ctx.cfg.get("screen", {}).copy()
    exp = ctx.cfg["export"]
    profiles = configured_profiles(ctx)
    cfg["train_on_what"], cfg["images"] = export_modes(exp)
    train_models = cfg.get("train_models")
    if not isinstance(train_models, list) or not train_models or not all(
        isinstance(m, str) and m.strip() for m in train_models
    ):
        raise ValueError("screen.train_models requires a nonempty list of model names")
    cfg["codex_reasoning"] = exp.get("codex_reasoning", "drop")
    if cfg["codex_reasoning"] not in ("drop", "summary"):
        raise ValueError("invalid export.codex_reasoning")
    allow_masked = cfg["train_on_what"] == "loss_mask"
    tokenizer = ctx.path(exp["tokenizer"])
    if cfg.get("hint_mode", "exclude") != "exclude":
        raise ValueError("invalid hint_mode")

    def build():
        counter = Counter(tokenizer)
        admitted = {r["task_id"] for r in ctx.rows("filtered") if r["keep"]}
        tasks = {r["task_id"]: r for r in ctx.rows("tasks")}
        directory = ctx.work / "segments"
        directory.mkdir()
        ledger = []
        for trial in ctx.rows("trials"):
            flags = set()
            fatal = set()
            details = {}
            total = 0
            within_limit = 0
            stripped = []
            if trial["task_id"] not in admitted:
                fatal.add("task_filtered")
            if trial["task_id"] and not trial["current_package"]:
                fatal.add("stale_package")
            if trial["reward"] != 1:
                fatal.add("not_passing")
            if not valid(trial):
                fatal.add("invalid_attempt")
            if trial["timed_out"]:
                if not cfg.get("allow_timeout", False) or re.search(
                    "setup|environment", str(trial["exception"]), re.I
                ):
                    fatal.add("timeout")
            elif not trial["clean_end"]:
                fatal.add("unclean_end")
            if trial["hint"]:
                fatal.add("hint_run")
            if len(trial["sessions"]) != 1:
                fatal.add("session_missing_or_ambiguous")
            path = digest(trial["trial_id"]) + ".json"
            if not fatal:
                try:
                    native = list(lines(Path(trial["sessions"][0])))
                    offline = tasks[trial["task_id"]]["offline"]
                    task_dirs = cfg.get("task_dirs", ["/app", "/work"])
                    raw_flags, raw_fatal, removed = inspect_omitted_events(
                        native, trial["harness"], offline, task_dirs
                    )
                    flags.update(raw_flags)
                    fatal.update(raw_fatal)
                    stripped.extend(sorted(removed))
                    profile = profiles.get(trial["harness"])
                    records, models, details = CONVERTERS[trial["harness"]](native, {**cfg, "profile": profile}, trial)
                    details["wire_context_hash"] = digest([profile, cfg["train_on_what"], cfg["codex_reasoning"], cfg["images"]])
                    observed = set(models) | set(trial["response_models"])
                    if trial["reported_model"]:
                        observed.add(trial["reported_model"])
                    details["observed_models"] = sorted(str(m) for m in observed)
                    if not models or not observed.issubset(train_models):
                        fatal.add("model_not_trainable")
                    has_tagged_hint = any(
                        b.get("hint") for rec in records for m in rec["messages"] for b in m["content"]
                    )
                    if has_tagged_hint or details.get("hint_run"):
                        fatal.add("hint_run")
                    affected = [list(image_dependents(r["messages"])) for r in records]
                    drop_all = cfg["images"] == "drop_trajectory" and any(affected)
                    details.update(image_rows=sum(map(bool, affected)), image_dropped_segments=[],
                                   image_masked_turns=0, image_dropped_trajectory=False)
                    for index, (record, dependent) in enumerate(zip(records, affected, strict=True)):
                        # Validate before and after editing so removing a call cannot mask corruption.
                        fatal.update(check(render(record, {}), allow_masked=allow_masked))
                        for m in dependent:
                            m["image_context"] = True
                        f, d, s = inspect_segment(
                            record,
                            tasks[trial["task_id"]]["offline"],
                            cfg.get("task_dirs", ["/app", "/work"]),
                        )
                        flags.update(f)
                        fatal.update(d)
                        stripped.extend(sorted(s))
                        if drop_all or (dependent and cfg["images"] == "drop_row"):
                            details["image_dropped_segments"].append(index)
                            continue
                        if dependent:
                            for m in image_dependents(record["messages"]):
                                details["image_masked_turns"] += m["loss"]
                                m["loss"] = 0
                            if not any(m["loss"] for m in record["messages"]):
                                details["image_dropped_segments"].append(index)
                                continue
                        wire = render(record, {})
                        fatal.update(check(wire, allow_masked=allow_masked))
                        length, supervised = counter.counts(wire)
                        if length <= exp.get("max_length", 194560):
                            total += supervised
                            within_limit += 1
                    if details["image_dropped_segments"]:
                        flags.add("image_content")
                        if len(details["image_dropped_segments"]) == len(records):
                            details["image_dropped_trajectory"] = True
                            fatal.add("image_content")
                    if not fatal:
                        # The trajectory judge still audits the full cleaned trajectory.
                        atomic(directory / path, records, sort_keys=False)
                except ProfileError:
                    raise
                except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                    fatal.add("conversion: " + str(exc))
            flags.update(fatal)
            disposition = "drop" if fatal else "strip" if stripped else "keep"
            ledger.append(
                dict(
                    trial_id=trial["trial_id"],
                    task_id=trial["task_id"],
                    harness=trial["harness"],
                    agent_version=trial["agent_version"],
                    disposition=disposition,
                    flags=sorted(flags),
                    reasons=sorted(fatal) or sorted(flags) or ["passed"],
                    stripped=sorted(set(stripped)),
                    within_limit=within_limit,
                    supervised_tokens=total,
                    source=None if fatal else path,
                    **details,
                )
            )
        ctx.emit("screen", ledger)
        ctx.emit("screen.ledger", ledger)
        return {"count": len(ledger), "dispositions": dict(Frequencies(r["disposition"] for r in ledger))}

    return ctx.stage(
        "screen",
        [ctx.dependency("filter-tasks"), cfg, digest(profiles), file_hash(tokenizer), exp.get("max_length", 194560)],
        build,
    )
