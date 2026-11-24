import pytest

from tbstrain.cli import main
from tbstrain.core import file_hash, read


@pytest.fixture
def stub(completed):
    cfg, out = completed
    script = cfg.parent / "fake-mint"
    script.write_text("""#!/bin/sh
set -eu
cd "$(dirname "$0")"
echo "$*" >> invocations
case "$1 $2" in
  'dataset upload') echo '{"dataset_version":"toy-v1","status":"available"}' ;;
  'dataset report') cat report.json ;;
  'train start') case " $* " in
    *" --dry-run "*) cat dry.json ;;
    *) echo '{"execution_id":"toy-training"}' ;;
    esac ;;
  'train show') echo '{"state":"queued"}' ;;
  'train events') echo '{"events":[{"name":"preflight.fields_ignored","fields":["loss_mask"]}]}' ;;
  *) exit 9 ;;
esac
""")
    (cfg.parent / "dry.json").write_text('{"config":{"train_on_what":"all_assistant_messages","max_length":194560}}')
    script.chmod(0o755)
    return cfg, out


def report(cfg, status="available", fatal="0", warn="0"):
    import json

    (cfg.parent / "report.json").write_text(
        json.dumps(dict(dataset_version="toy-v1", status=status, fatal_count=fatal, warn_count=warn))
    )


def test_submit(stub):
    cfg, out = stub
    report(cfg)
    args = ["submit", "--config", str(cfg)]
    assert main(args) == 0
    log = cfg.parent / "invocations"
    commands = log.read_text().splitlines()
    assert len(commands) == 3 and "--dry-run" in commands[-1]
    assert file_hash(out / "train.jsonl") in commands[0] and "--timeout 30m" in commands[0]
    assert main(args + ["--confirm"]) == 0
    commands = log.read_text().splitlines()
    assert (
        "train start " in commands[-3]
        and "--dry-run" not in commands[-3]
        and "--idempotency-key" in commands[-3]
    )
    assert "train show toy-training" in commands[-2] and "train events toy-training" in commands[-1]
    assert read(out / "reports/submit.json")["loss_mask_ignored_observed"]
    assert len(list((out / "submit-responses").rglob("*-*.json"))) == 9
    assert main(args + ["--confirm"]) == 0 and log.read_text().splitlines() == commands


@pytest.mark.parametrize(
    "status,fatal,warn,override,accepted",
    [
        ("available", "1", "0", False, False),
        ("available", "1", "0", True, False),
        ("available", "0", "1", False, False),
        ("available", "0", "1", True, True),
        ("processing", "0", "0", True, False),
    ],
)
def test_report_gate(stub, status, fatal, warn, override, accepted):
    cfg, out = stub
    report(cfg, status, fatal, warn)
    args = ["submit", "--config", str(cfg), "--confirm"]
    if override:
        args.append("--allow-warn")
    assert (main(args) == 0) == accepted
    assert ("train start" in (cfg.parent / "invocations").read_text()) == accepted
    assert list((out / "submit-responses").rglob("*-report.json"))


@pytest.mark.parametrize("case,reason", [("masked", "masked assistant"), ("length", "max_length"),
                                         ("missing", "train_on_what")])
def test_confirm_refuses_unsafe_dry_run(stub, case, reason):
    from tbstrain.core import atomic, lines, write_lines
    from tbstrain.wire import render

    cfg, out = stub
    report(cfg)
    effective = dict(train_on_what="all_assistant_messages", max_length=194560)
    if case == "masked":
        cfg.write_text(cfg.read_text().replace("[export]", '[export]\ntrain_on_what="loss_mask"'))
        sources = list(lines(out / "train.sources.jsonl"))
        source = next(s for s in sources if sum(m["role"] == "assistant" for m in s["segment"]["messages"]) > 1)
        next(m for m in source["segment"]["messages"] if m["role"] == "assistant")["loss"] = 0
        write_lines(out / "train.sources.jsonl", sources)
        write_lines(out / "train.jsonl", [render(s["segment"], s["meta"]) for s in sources])
    elif case == "length":
        effective["max_length"] = 1
    else:
        del effective["train_on_what"]
    atomic(cfg.parent / "dry.json", {"config": effective})
    assert main(["submit", "--config", str(cfg), "--confirm"]) == 1
    assert reason in read(out / "reports/submit.json")["error"]
    commands = (cfg.parent / "invocations").read_text().splitlines()
    assert len(commands) == 3 and "--dry-run" in commands[-1]
