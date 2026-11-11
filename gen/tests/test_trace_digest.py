"""review.trace_digest must survive every stream-json event shape Claude Code writes (e2e-batch2: a
`system/permission_denied` event carries `message` as a string, which crashed S9R bundle building)."""
import json

from tbsgen import review


def test_trace_digest_tolerates_string_message_and_non_dict_lines(tmp_path):
    t = tmp_path / "trial"
    (t / "agent").mkdir(parents=True)
    rows = [
        {"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "hello"},
                                                                 {"type": "tool_use", "id": "t1", "name": "Bash",
                                                                  "input": {"command": "ls"}}]}},
        {"type": "system", "subtype": "permission_denied", "tool_name": "Bash", "tool_use_id": "t2",
         "decision_reason": "Dangerous rm operation", "message": "Dangerous rm operation detected"},
        {"type": "system", "subtype": "other", "message": "a plain string"},
        ["not", "a", "dict"],
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "result": "done"},
    ]
    (t / "agent" / "claude-code.txt").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    d = review.trace_digest(t)
    assert "hello" in d and "Bash: ls" in d and "(denied) Bash: Dangerous rm operation" in d
    assert "subtype=success" in d and "- ok" in d


def test_trace_digest_tolerates_odd_nested_fields(tmp_path):
    t = tmp_path / "trial"
    (t / "agent").mkdir(parents=True)
    rows = [
        {"type": "assistant", "message": {"id": ["x"], "content": [
            {"type": "text", "text": None}, {"type": "tool_use", "id": ["y"], "name": "Bash", "input": "ls -la"},
            {"type": "tool_use", "id": "z", "name": "Edit", "input": ["a"]}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": 42},
                                                 {"type": "tool_result", "content": [{"text": None}, "s"]}]}},
    ]
    (t / "agent" / "claude-code.txt").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    d = review.trace_digest(t)
    assert 'Bash: "ls -la"' in d and "Edit:" in d and "- 42" in d
