"""PKG states the verifier time limit for PROG tasks (e2e-batch2 review: the 780 s scored cap was unstated)."""
from tbsgen import package, runner
from tbsgen.config import load_config


def test_prog_statement_gets_the_time_limit_once():
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\nDo it.\n")
    assert f"{runner.scored_budget(cfg):g} seconds" in t1 and package.TIME_NOTE_MARK in t1
    assert f"on {package.TASK_CPUS} CPU cores" in t1 and "once on each hidden input set" in t1
    assert package.instruction_text(cfg, {"form": "PROG"}, t1) == t1          # idempotent
    assert t1.count("## Verifier time limit") == 1


def test_non_prog_statement_unchanged_and_old_note_removed():
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\n")
    assert package.instruction_text(cfg, {"form": "FILE"}, t1) == "# Task\n"


def test_text_after_the_marked_block_is_kept():
    """Codex r1: a marked statement copied back into task/ and then revised must keep the revision's text."""
    cfg = load_config(None)
    t1 = package.instruction_text(cfg, {"form": "PROG"}, "# Task\n")
    revised = t1 + "\n## Pass criteria\n- `m` <= 0.5\nRun `python /app/output/solution_cli.py <input_dir> <output_dir>`\n"
    t2 = package.instruction_text(cfg, {"form": "PROG"}, revised)
    assert "## Pass criteria" in t2 and "solution_cli.py <input_dir> <output_dir>" in t2
    assert t2.count("## Verifier time limit") == 1


def test_fractional_budget_kept_and_task_toml_cpus_match():
    cfg = load_config(None, {"package": {"verifier_overhead_sec": 120.5}})
    assert "779.5 seconds" in package.instruction_text(cfg, {"form": "PROG"}, "x\n")
    assert f"cpus = {package.TASK_CPUS}\n" in package.TASK_TOML.format(
        task_id="t", desc="d", keywords="[]", domain="", field="", form="", expert_h=1.0, verifier_sec=1.0,
        agent_sec=1.0, cpus=package.TASK_CPUS, verifier_network="", agent_network="", env_network="")
