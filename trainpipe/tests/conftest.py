"""All tasks, scientific claims, sessions and responses here are hand-written fixtures."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from tbstrain.core import atomic, dumps, write_lines
from tbstrain.cli import main


def block(text):
    return {"type": "text", "text": text}


def cc_message(role, content, **attrs):
    msg = {"content": content}
    if role == "assistant":
        msg["model"] = "student-model"
    return {"type": role, "message": msg, **attrs}


def toy_tool(name="sample"):
    return dict(name=name, description="Measure the toy sample.",
                input_schema={"type": "object", "properties": {"n": {"type": "integer"}}})


def toy_profile():
    return dict(version="1", system=[block("Synthetic instructions.")],
                tools=[toy_tool(), toy_tool("Bash"), toy_tool("unused")])


def link_events(events):
    parent = None
    for i, event in enumerate(events):
        uid = event.setdefault("uuid", f"node-{i}")
        event.update(parentUuid=parent, version="1")
        parent = uid
    return events


def claude_events():
    a = cc_message("assistant", [{"type": "thinking", "thinking": "Compare the two synthetic samples."}])
    a["message"]["id"] = "response-a"
    calls = cc_message(
        "assistant",
        [{"type": "tool_use", "name": "sample", "id": f"sample-{n}", "input": {"n": n}} for n in (1, 2)],
    )
    calls["message"]["id"] = "response-a"
    events = [
        cc_message("user", "Estimate a synthetic slope."),
        a,
        calls,
        cc_message(
            "user", [{"type": "tool_result", "tool_use_id": f"sample-{n}", "content": str(n)} for n in (1, 2)]
        ),
        {"type": "attachment", "rendered": [{"content": "Synthetic context limit reached."}]},
        {
            "type": "system",
            "subtype": "compact_boundary",
            "compactMetadata": {"preservedMessages": {"allUuids": ["old"]}},
        },
        cc_message("user", "Summary:\nTwo samples are available.", isCompactSummary=True),
        cc_message("assistant", [block("Replay should disappear.")], uuid="old", isSidechain=True),
        cc_message(
            "assistant",
            [{"type": "tool_use", "id": "query", "name": "Bash", "input": {"command": "ls /tests"}}],
        ),
        cc_message(
            "user",
            [{"type": "tool_result", "tool_use_id": "query", "content": "No files found", "is_error": True}],
        ),
        cc_message("assistant", [block("The synthetic slope is one.")]),
    ]
    link_events(events)
    boundary = events[5]
    boundary.update(logicalParentUuid=events[4]["uuid"], parentUuid=None)
    boundary["compactMetadata"] = {"preservedMessages": {"anchorUuid": events[6]["uuid"]}}
    events[8]["parentUuid"] = events[6]["uuid"]
    return events


def codex_events():
    def item(kind, **fields):
        return {"type": "response_item", "payload": {"type": kind, **fields}}

    return [
        {"type": "session_meta", "payload": {"base_instructions": {"text": "Use the synthetic samples."}}},
        {"type": "turn_context", "payload": {"model": "student-model", "tools": [toy_tool()]}},
        item("message", role="user", content=[{"type": "input_text", "text": "Estimate a synthetic slope."}]),
        item("function_call", name="sample", call_id="a", arguments='{"n":2}'),
        {"type": "event_msg", "payload": {"type": "token_count"}},
        item("function_call_output", call_id="a", output="2"),
        item(
            "message",
            role="assistant",
            content=[{"type": "output_text", "text": "The synthetic slope is one."}],
        ),
        {"type": "event_msg", "payload": {"type": "token_count"}},
    ]


def pi_events():
    def event(role, content, **extra):
        return dict(
            type="event",
            turn=1,
            event={"type": "message_end", "message": {"role": role, "content": content, **extra}},
        )

    return [
        {"type": "turn_start", "turn": 1},
        event("user", "Estimate a synthetic slope."),
        {
            "type": "request",
            "turn": 1,
            "body": {
                "model": "student-model",
                "tools": [],
                "messages": [{"role": "system", "content": "Use the samples."}],
            },
        },
        event("assistant", [block("The synthetic slope is one.")], stopReason="stop", model="student-model"),
        {"type": "turn_end", "turn": 1},
        {"type": "turn_start", "turn": 2},
        event("user", "An unfinished follow-up."),
    ]


@pytest.fixture
def workspace(tmp_path):
    run = tmp_path / "run"
    (run / "release").mkdir(parents=True)
    manifests = []
    for tier, harness, events, n in [
        ("easy", "pi", pi_events(), 3),
        ("medium", "codex", codex_events(), 2),
        ("hard", "claude-code", claude_events(), 1),
    ]:
        item = run / "items" / tier
        pkg = item / "pkg"
        (pkg / "tests").mkdir(parents=True)
        (pkg / "instruction.md").write_text("Estimate a synthetic slope.")
        (pkg / "tests/test_outputs.py").write_text("assert measured_slope == 1")
        atomic(item / "state.json", {"stage": "S10", "status": "done"})
        atomic(
            item / "_spec/spec.json",
            {"field": "toy field", "domain": "simulation", "metrics": {"slope": "exact"}},
        )
        cal = [
            dict(
                who="student",
                model="student-model",
                job_name=f"{tier}-{i}",
                trial={"trial": "one"},
                reward=int(i < n),
                outcome="pass" if i < n else "fail",
                result_type="pass" if i < n else "fail_delivered",
            )
            for i in range(3)
        ]
        cal += [
            dict(who="teacher", model="teacher-model", job_name=tier + "-teacher", trial={"trial": "one"},
                 reward=1, outcome="pass")
        ]
        atomic(item / "calib/runs.json", cal)
        manifests.append(
            dict(
                item_id=tier,
                title=tier + " synthetic exercise",
                calib_runs=cal,
                label="released",
                review={"verdict": "pass"},
            )
        )
        trial = run / "calib-jobs" / (tier + "-0") / "one"
        native = {
            "pi": "agent/pi/sessions/tiny.jsonl",
            "claude-code": "agent/sessions/projects/toy/tiny.jsonl",
            "codex": "agent/sessions/2025/01/01/rollout-toy.jsonl",
        }[harness]
        session = trial / native
        session.parent.mkdir(parents=True)
        write_lines(session, events)
        atomic(
            trial / "config.json",
            {"agent": {"name": harness, "model_name": "student-model"}, "task": {"path": str(pkg)}},
        )
        atomic(
            trial / "result.json",
            {
                "id": tier + "-trial",
                "task_name": tier,
                "finished_at": "2025-01-01T00:01:00",
                "started_at": "2025-01-01T00:00:00",
                "verifier_result": {"rewards": {"reward": 1}},
                "agent_info": {"version": "1", "model_info": {"name": "student-model"}},
            },
        )
    write_lines(run / "release/manifest.jsonl", manifests)
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "compact.txt").write_text("Summarize the synthetic observations.")
    from tbstrain.profile import extract
    capture = tmp_path / "capture.json"
    atomic(capture, toy_profile())
    extract([capture], tmp_path / "wire-profile", "1")
    cfg = tmp_path / "pipeline.toml"
    cfg.write_text("""out = "out"
[ingest]
runs = ["run"]
[screen]
train_models = ["student-model"]
[judge]
model = "judge-model"
base_url = "https://judge.example/v1"
concurrency = 128
api_key_env = "UNUSED"
backoff = 0
[select]
mode = "shares"
max_token_share = 1.0
shares = { hard=1, medium=1, easy=1 }
[export]
tokenizer = "tokenizer.json"
compaction_instruction = "compact.txt"
[export.wire.claude-code]
dir = "wire-profile"
[rollout]
harness = "pi"
model = "student-model"
attempts = 2
concurrency = 4
job_name = "more"
jobs_dir = "more-jobs"
environment = { import_path = "example:Environment" }
agents = { pi = { import_path="example:Pi", kwargs={ effort="high" } } }
[submit]
cli = "./fake-mint"
base_model = "student-model"
config_file = "train.toml"
""")
    (tmp_path / "train.toml").write_text('train_on_what = "all_assistant_messages"')
    return cfg


class FakeJudge:
    def __init__(self):
        self.calls = []

    async def __call__(self, url, headers, body, timeout):
        assert url == "https://judge.example/v1/messages"
        self.calls.append(body)
        if "grader and metric" in body["system"]:
            result = {
                "verdict": "keep",
                "difficulty": "medium",
                "rationale": "The metric matches the requested slope.",
                "evidence": ["Estimate a synthetic slope."],
            }
        else:
            result = {
                "quality": 5,
                "findings": [],
                "wasted_turn_ratio": 0.0,
                "rationale": "The estimate uses the samples.",
            }
        return {"content": [block(dumps(result))], "usage": {"input_tokens": 20, "output_tokens": 10}}


@pytest.fixture
def judge_stub():
    return FakeJudge()


@pytest.fixture
def completed(workspace, judge_stub):
    assert main(["run", "--config", str(workspace)], transport=judge_stub) == 0
    return workspace, workspace.parent / "out"
