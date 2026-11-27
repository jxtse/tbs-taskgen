"""Fixtures, not tests: the stage-contract fixtures contain verifier files (tests/test_outputs.py) that pytest must
not collect — they are data for `test_stage_contracts.py`, run against /app/output inside a task image."""
collect_ignore_glob = ["contracts/*"]
