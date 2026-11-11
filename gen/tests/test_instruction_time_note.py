"""PKG appends the verifier time limit for PROG tasks (e2e-batch2 review: the 780 s scored cap was unstated)."""
from tbsgen import package, runner
from tbsgen.config import load_config


def test_prog_statement_gets_the_time_limit_once():
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\nDo it.\n")
    assert f"{int(runner.scored_budget(cfg))} seconds" in t1 and package.TIME_NOTE_MARK in t1
    assert package.instruction_text(cfg, {"form": "PROG"}, t1) == t1          # idempotent
    assert t1.count("## Verifier time limit") == 1


def test_non_prog_statement_unchanged_and_old_note_removed():
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\n")
    assert package.instruction_text(cfg, {"form": "FILE"}, t1) == "# Task\n"
