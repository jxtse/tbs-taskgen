"""Official-task (o70) quota before a paper's first S5G authoring job."""
import json
from types import SimpleNamespace

from tbsgen import diversity, stages_a
from tbsgen.config import load_config
from tbsgen.state import Store, write_json


def _setup(tmp_path, inv_extra=None):
    run = tmp_path / "run"
    store = Store(run)

    def item(id, target, status="active"):
        it = store.item(id)
        it.state["status"] = status
        it.save()
        write_json(it.p("S1_candidate.json"), {"title": f"{id} work", "_target": target})
        return it

    item("r1", "method:m-a", "done")
    item("r2", "method:m-a", "done")
    item("r3", "method:m-b", "done")
    (run / "release").mkdir()
    (run / "release/manifest.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"item_id": "r1", "label": "hard"},
        {"item_id": "r2", "label": "hard"},
        {"item_id": "r2", "withdrawn": True},          # withdrawn: not counted
        {"item_id": "r3", "label": "unstable"},        # not hard: not counted
    ]) + "\n")
    inv = {"target_group": {"method:m-a": "o70-x", "method:m-b": "o70-x", "task:o70-y": "o70-y"},
           "library": {"o70-x": {"hard": 2, "medium": 1, "easy": 5}, "o70-y": {"hard": 0}},
           "library_items": []}
    inv.update(inv_extra or {})
    write_json(tmp_path / "inv.json", inv)
    P = SimpleNamespace(cfg=load_config(), store=store)
    P.cfg["diversity"].update(o70_inventory=str(tmp_path / "inv.json"), o70_max_hard=4)
    return P, item


def test_o70_quota_counts_library_and_unadmitted_hard_releases(tmp_path):
    P, item = _setup(tmp_path)
    c = item("c", "method:m-b")
    q = diversity.o70_quota(P, c)
    assert q["group"] == "o70-x" and q["library_hard"] == 2 and q["released_hard"] == ["r1"]
    assert q["hard"] == 3 and q["reason"] is None
    P.cfg["diversity"]["o70_max_hard"] = 3
    assert diversity.o70_quota(P, c)["reason"] == "diversity:o70_full:o70-x"
    # a release the library already counts is not counted twice
    P2, item2 = _setup(tmp_path / "b", {"library_items": ["r1"]})
    assert diversity.o70_quota(P2, item2("c", "method:m-b"))["hard"] == 2
    # unmapped target or unreadable inventory never blocks
    assert diversity.o70_quota(P, item("u", "method:unknown"))["reason"] is None
    (tmp_path / "inv.json").write_text("{not json")
    assert diversity.o70_quota(P, c)["reason"] is None
    P.cfg["diversity"]["o70_inventory"] = None
    assert diversity.o70_quota(P, c) is None


def test_o70_enforcement_is_separate_and_first_authoring_only(tmp_path):
    P, item = _setup(tmp_path)
    P.cfg["diversity"].update(o70_max_hard=3, max_per_template=99, max_area_share=1.0)
    c = item("c", "method:m-a")
    # observe only: recorded, not enforced
    assert stages_a._agent_admission(P, c, "S5G") is None
    rec = c.state["diversity"]["S5G"]["o70"]
    assert rec["reason"] == "diversity:o70_full:o70-x" and rec["enforce"] is False
    # o70_enforce alone blocks, even with the area/template rules off
    P.cfg["diversity"]["o70_enforce"] = True
    assert P.cfg["diversity"]["enforce"] is False
    why = stages_a._agent_admission(P, c, "S5G")
    assert why == "diversity:o70_full:o70-x" and stages_a._admission_status(why) == "parked"
    # a group with room passes
    assert stages_a._agent_admission(P, item("y", "task:o70-y"), "S5G") is None
    # re-authoring after a finished S5G job and hardening rounds are not re-judged
    write_json(c.p("S5G_result.json"), {"outcome": "ok"})
    assert diversity.admission(P, c, "S5G") is None
    h = item("h", "method:m-a")
    h.state["harden_active"] = True
    assert diversity.admission(P, h, "S5G") is None
    # S10 never applies the o70 quota
    assert "o70" not in (diversity.admission(P, item("s", "method:m-a"), "S10") or "")
