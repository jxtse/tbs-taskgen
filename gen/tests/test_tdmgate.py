"""TDM verbatim gate (tdmgate.py): unit tests.  The integration test (test_integration.py) runs it inside S7."""
import json

from tbsgen import tdmgate
from tbsgen.config import load_config

SRC = ("Introduction. We consider the estimation of decay constants in detector afterglow traces corrupted by "
       "one-sided spikes, and we propose a weighted log-linear estimator whose weights follow the squared signal, "
       "followed by an iterative rejection of positive outliers until convergence of the fitted rate parameter. ") * 3


def mk_item(tmp_path, channel="elsevier", cached=None, instruction="Fit the decay constants.", text=SRC):
    item = tmp_path / "item"
    task = item / "task"
    data = tmp_path / "gen0" / "environment" / "data"
    task.mkdir(parents=True)
    data.mkdir(parents=True)
    (task / "instruction.md").write_text(instruction)
    rec = dict(source="cache", cached_channel=cached) if cached else dict(source=channel)
    (item / "S3_fulltext.json").write_text(json.dumps(rec))
    if text is not None:
        (item / "fulltext.txt").write_text(text)
    return item, task, data


def test_longest_overlap_is_whitespace_and_case_insensitive():
    a = "Intro:  WE   consider\nthe estimation of decay constants in detector afterglow traces corrupted by"
    n, snip = tdmgate.longest_overlap(a, SRC, 20)
    assert n == len(tdmgate.normalise("we consider the estimation of decay constants in detector afterglow "
                                       "traces corrupted by"))
    assert snip.startswith("we consider")
    assert tdmgate.longest_overlap("short", SRC, 20) == (0, "")


def test_overlapping_repeats_are_found_exactly():
    """A rolling scan that skips ahead after a hit misses overlapping occurrences (Codex review r1)."""
    c = "b" * 198
    public, source = "x a " + c + " z", "y a " + c + " q#a " + c + " z"
    assert tdmgate.longest_overlap(public, source)[0] == len(("a " + c + " z").strip())


def test_matches_brute_force_on_random_texts():
    import random
    rnd = random.Random(7)
    for _ in range(200):
        a = "".join(rnd.choice("ab ") for _ in range(rnd.randint(0, 40)))
        b = "".join(rnd.choice("ab ") for _ in range(rnd.randint(0, 40)))
        na, nb = tdmgate.normalise(a), tdmgate.normalise(b)
        best = max([len(na[i:j].strip()) for i in range(len(na)) for j in range(i + 1, len(na) + 1)
                    if na[i:j] in nb] or [0])
        assert tdmgate.longest_overlap(a, b)[0] == best, (a, b)


def test_large_and_unknown_suffix_files_are_checked(tmp_path):
    cfg = load_config()
    item, task, data = mk_item(tmp_path)
    (data / "big.csv").write_text("x," * 1_100_000 + "\n" + SRC)            # > 2 MB
    (data / "notes.dat").write_text(SRC)                                    # unknown suffix, textual
    (data / "blob.bin").write_bytes(b"\0\1\2" + SRC.encode())              # binary: skipped
    r = tdmgate.check(cfg, item, task, data)
    assert {f["file"] for f in r["findings"]} == {"environment/data/big.csv", "environment/data/notes.dat"}


def test_symlinked_file_and_dir_are_followed(tmp_path):
    """Packaging copies link targets, so the gate must read them too (Codex r2)."""
    cfg = load_config()
    item, task, data = mk_item(tmp_path)
    outside = tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    (outside / "quote.txt").write_text(SRC)
    (outside / "sub" / "more.txt").write_text(SRC)
    (data / "notes.txt").symlink_to(outside / "quote.txt")
    (data / "linked").symlink_to(outside / "sub", target_is_directory=True)
    (data / "loop").symlink_to(data, target_is_directory=True)        # a cycle must not hang
    r = tdmgate.check(cfg, item, task, data)
    assert {f["file"] for f in r["findings"]} >= {"environment/data/notes.txt", "environment/data/linked/more.txt"}


def test_quote_up_to_200_chars_passes_longer_blocks(tmp_path):
    cfg = load_config()
    quote = SRC[14:14 + 200]
    item, task, data = mk_item(tmp_path, instruction=f"# Task\n> {quote}\nDo the fit.")
    r = tdmgate.check(cfg, item, task, data)
    assert r["applies"] and r["ok"], r
    item, task, data = mk_item(tmp_path / "b", instruction=f"# Task\n{SRC[14:14 + 201]}\nDo the fit.")
    r = tdmgate.check(cfg, item, task, data)
    assert not r["ok"] and r["findings"][0]["file"] == "instruction.md" and r["findings"][0]["chars"] == 201


def test_data_files_are_checked_too(tmp_path):
    cfg = load_config()
    item, task, data = mk_item(tmp_path, channel="wiley")
    (data / "README.txt").write_text("Notes: " + SRC[:400])
    r = tdmgate.check(cfg, item, task, data)
    assert not r["ok"] and r["findings"][0]["file"] == "environment/data/README.txt"


def test_cache_hit_uses_the_channel_that_filled_the_cache(tmp_path):
    cfg = load_config()
    item, task, data = mk_item(tmp_path, cached="elsevier", instruction=SRC)
    assert not tdmgate.check(cfg, item, task, data)["ok"]


def test_oa_and_unknown_channels_are_not_checked_by_default(tmp_path):
    cfg = load_config()
    for ch in ("oa", "browser"):
        item, task, data = mk_item(tmp_path / ch, channel=ch, instruction=SRC)
        assert tdmgate.check(cfg, item, task, data) == dict(channel=ch, applies=False, max_chars=200, findings=[],
                                                            ok=True)
    item, task, data = mk_item(tmp_path / "u", channel=None, instruction=SRC)
    assert tdmgate.check(cfg, item, task, data)["applies"] is False
    cfg["gates"]["tdm_verbatim_unknown"] = True
    assert tdmgate.check(cfg, item, task, data)["ok"] is False


def test_purged_tdm_text_fails_closed(tmp_path):
    item, task, data = mk_item(tmp_path, text=None)
    r = tdmgate.check(load_config(), item, task, data)
    assert r["applies"] and not r["ok"] and "purged" in r["error"]


def test_gate_is_blocking_by_default():
    assert "tdm_verbatim" in load_config()["gates"]["blocking"]
