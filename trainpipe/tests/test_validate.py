import copy

import pytest

from conftest import toy_tool
from tbstrain.cli import main
from tbstrain.core import dumps, lines
from tbstrain.native import message, segment
from tbstrain.wire import check, render


@pytest.fixture
def sample():
    record = segment(
        [
            message("system", "Synthetic instructions"),
            message(
                "assistant",
                [{"type": "tool_use", "name": "inspect", "id": f"id-{n}", "input": {"n": n}} for n in (1, 2)],
            ),
            message(
                "user", [{"type": "tool_result", "tool_use_id": f"id-{n}", "content": str(n)} for n in (1, 2)]
            ),
            message("assistant", "A synthetic answer"),
        ], [toy_tool("inspect")]
    )
    return render(record, {})


@pytest.mark.parametrize(
    "path,value,rule",
    [
        (["messages", 0, "content"], 42, "content_type"),
        (["messages", 1, "model"], "undocumented", "assistant_keys"),
        (["messages", 1, "reasoning_content"], [], "reasoning_type"),
        (["messages", 1, "tool_calls", 1, "id"], "id-1", "tool_id"),
        (["messages", 1, "tool_calls", 1, "id"], "", "tool_id"),
        (["messages", 1, "tool_calls", 1, "type"], "custom", "tool_call_schema"),
        (["messages", 1, "tool_calls", 0, "function", "arguments"], {}, "arguments"),
        (["messages", 1, "tool_calls", 0, "function", "arguments"], "{", "arguments"),
        (["messages", 1, "tool_calls", 0, "function", "arguments"], '{"x":NaN}', "arguments"),
        (["messages", 1, "tool_calls", 0, "function", "name"], "absent", "missing_tool_schema"),
        (["tools", 0, "type"], "other", "tools_schema"),
        (["messages", 2, "tool_call_id"], "id-2", "tool_order"),
        (["messages", 2, "role"], "user", "tool_response_missing"),
        (["messages", 2, "name"], "inspect", "tool_keys"),
        (["messages", 0, "role"], "tool", "first_role"),
        (["messages", -1, "role"], "user", "no_final_answer"),
        (["messages", -1, "content"], " ", "no_final_answer"),
        (["messages", 1, "tool_calls"], [], "empty_assistant"),
        (["messages", 0, "role"], "developer", "role"),
        (["loss_mask"], [0, 1], "mask_length"),
        (["loss_mask", 0], 1, "invalid_loss"),
        (["loss_mask", 1], True, "invalid_loss"),
        (["loss_mask", 1], 0, "masked_assistant"),
        (["meta"], None, "meta"),
    ],
)
def test_negative_invariant(sample, path, value, rule):
    assert check(sample) == []
    container = sample
    for k in path[:-1]:
        container = container[k]
    container[path[-1]] = value
    assert rule in check(sample)


def test_missing_response_and_stray_tool(sample):
    del sample["messages"][3]
    del sample["loss_mask"][3]
    assert "tool_response_missing" in check(sample)
    sample["messages"].insert(1, {"role": "tool", "tool_call_id": "x", "content": "ignored"})
    assert "tool_order" in check(sample)


def test_masked_assistant_override():
    src = segment(
        [
            message("system", "Instructions"),
            message("assistant", "History"),
            message("user", "Next"),
            message("assistant", "Answer"),
        ]
    )
    src["messages"][1]["loss"] = 0
    row = render(src, {})
    assert check(row) == ["masked_assistant"]
    assert check(row, allow_masked=True) == []


@pytest.mark.parametrize(
    "content,reason", [(b"\xff", "utf8"), (b"NaN", "non-finite JSON"), (b"", "empty_dataset")]
)
def test_validate_file_errors(workspace, content, reason):
    path = workspace.parent / "broken.jsonl"
    path.write_bytes(content)
    assert main(["validate", str(path), "--config", str(workspace)]) == 1
    ledger = list(lines(workspace.parent / "out/validate.ledger.jsonl"))
    assert reason in str(ledger[0]["errors"])


def test_max_length_source_and_cli_override(workspace):
    src = segment(
        [
            message("system", "Instructions"),
            message("assistant", "History"),
            message("user", "Next"),
            message("assistant", "Answer"),
        ]
    )
    src["messages"][1]["loss"] = 0
    src["messages"][1]["response_id"] = "history"
    src["messages"][3]["response_id"] = "answer"
    path = workspace.parent / "custom.jsonl"
    source = path.with_suffix(".sources.jsonl")
    source.write_text(dumps({"meta": {}, "segment": src}) + "\n")
    row = render(src, {})
    path.write_text(dumps(row) + "\n")
    args = ["validate", str(path), "--config", str(workspace)]
    assert main(args) == 1
    args.append("--allow-masked-assistant")
    assert main(args) == 0
    changed = copy.deepcopy(row)
    changed["messages"][0]["content"] = "Different input"
    path.write_text(dumps(changed) + "\n")
    assert main(args) == 1
    ledger = workspace.parent / "out/validate.ledger.jsonl"
    assert "provenance" in next(lines(ledger))["errors"]
    path.write_text(dumps(row) + "\n")
    workspace.write_text(workspace.read_text().replace("[export]", "[export]\nmax_length=1"))
    assert main(args) == 1
    assert "max_length" in next(lines(ledger))["errors"]


def test_validate_missing_provenance(workspace, sample):
    path = workspace.parent / "custom.jsonl"
    path.write_text(dumps(sample) + "\n")
    assert main(["validate", str(path), "--config", str(workspace)]) == 1
    assert "source_missing" in next(lines(workspace.parent / "out/validate.ledger.jsonl"))["errors"]


@pytest.mark.parametrize("partial", [False, True])
def test_validate_legacy_response_boundaries(workspace, partial):
    src = segment([message("system", "Instructions"), message("assistant", "First answer"),
                   message("user", "Next tile"), message("assistant", "Second answer")])
    if partial:
        src["messages"][1]["response_id"] = "first"
    row = render(src, {})
    path = workspace.parent / "legacy.jsonl"
    path.write_text(dumps(row) + "\n")
    path.with_suffix(".sources.jsonl").write_text(dumps(dict(meta={}, segment=src)) + "\n")
    args = ["validate", str(path), "--config", str(workspace)]
    ledger = workspace.parent / "out/validate.ledger.jsonl"
    assert main(args) == 1
    assert next(lines(ledger))["errors"] == ["unverifiable_response_boundaries"]
    args.append("--allow-legacy-provenance")
    assert main(args) == 0
    entry = next(lines(ledger))
    assert entry["reason"] == "warning" and entry["warnings"] == ["unverifiable_response_boundaries"]
    row["messages"][-1]["content"] = "Different answer"
    path.write_text(dumps(row) + "\n")
    assert main(args) == 1
    assert next(lines(ledger))["errors"] == ["provenance"]


def test_validate_split_native_response(workspace):
    path = workspace.parent / "split.jsonl"
    source = segment([message("system", "Instructions")], [toy_tool()])
    for tid in ("a", "b"):
        source["messages"].extend([
            {**message("assistant", [dict(type="tool_use", id=tid, name="sample", input={})]),
             "response_id": "batch"},
            message("user", [dict(type="tool_result", tool_use_id=tid, content="ok")]),
        ])
    source["messages"].append(message("assistant", "Done."))
    row = render(source, {})
    assert not check(row)
    path.write_text(dumps(row) + "\n")
    path.with_suffix(".sources.jsonl").write_text(dumps(dict(meta={}, segment=source)) + "\n")
    assert main(["validate", str(path), "--config", str(workspace)]) == 1
    assert next(lines(workspace.parent / "out/validate.ledger.jsonl"))["errors"] == ["split_native_response"]


@pytest.mark.parametrize("mode", ["drop_row", "mask_after"])
def test_validate_rejects_supervision_after_image(workspace, mode):
    source = segment([message("system", "Instructions"), message("assistant", "Ready."),
                      message("user", [{"type": "image"}]), message("assistant", "Describe the tile.")])
    row = render(source, {})
    path = workspace.parent / "image.jsonl"
    path.write_text(dumps(row) + "\n")
    path.with_suffix(".sources.jsonl").write_text(dumps(dict(meta={}, segment=source)) + "\n")
    workspace.write_text(workspace.read_text().replace("[export]", f'[export]\nimages="{mode}"'))
    assert main(["validate", str(path), "--config", str(workspace)]) == 1
    assert next(lines(workspace.parent / "out/validate.ledger.jsonl"))["errors"] == ["image_content"]
