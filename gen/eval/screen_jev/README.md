# S2 screen evaluation: rules vs TypeSafe Jev

Question: does the Jev screen (`models.roles.screen = "jev"`) keep the papers that are worth turning into
computational tasks better than the keyword rules, at a speed and cost that still fit S2?

## What the labels measure

| Label source | What it measures | Used for |
|---|---|---|
| Reference labels from a teacher model (Claude Opus) following [`RUBRIC.md`](RUBRIC.md), written before any screen was run | "Suitable for a computational task" from the record alone: computational core, numerical check, data available or simulatable, single-machine scale | Main metrics. **Model labels, not human gold labels.** 74% keep; 40% of labels marked low confidence |
| Downstream outcome of an earlier pipeline run (whether the paper became an accepted task) | Real downstream success, but only on papers that an earlier screen had already selected (almost all positives) | Recall cross-check that does not depend on any model label |
| Human topic-relevance labels from the retrieval evaluation (0/1/2) | Relevance to one retrieval target | Sampling strata only: relevance to a topic is not suitability for a task |

Reference set: 200 abstracts sampled by stratum (end-to-end papers, accepted / failed papers of the earlier
pipeline, retrieval-eval papers by relevance label, candidates the earlier pipeline's screen excluded or scored
low/high, papers without an abstract screened on the full-text opening). Split by hash of the DOI into a dev half
(90) and a test half (110). Thresholds and rule changes were chosen on dev only; numbers below are from test.

## Results (test half, n = 110, reference labels)

| Screen | Recall | Precision | Keep rate | TP / FP / FN / TN |
|---|---|---|---|---|
| rules-v1 (previous) | 0.81 | 0.84 | 0.70 | 65 / 12 / 15 / 18 |
| rules-v2 (stems once, survey fix, `min_score` 1) | 0.94 | 0.78 | 0.87 | 75 / 21 / 5 / 9 |
| **Jev** (`jev-1.14.0`, rule `jev-r2`) | **0.975** | **0.90** | 0.79 | 78 / 9 / 2 / 21 |

- Bootstrap 95% intervals (test half): Jev recall 0.94–1.00 vs rules-v1 0.73–0.90 (difference +0.07 to +0.26);
  precision Jev 0.83–0.96 vs rules-v1 0.77–0.92 (difference −0.02 to +0.12, not significant on 110 papers).
- High-confidence reference labels only (68 papers): Jev recall 1.00, precision 0.98; rules-v1 0.88 / 0.92.
- Jev verdicts on test: 73 keep, 14 uncertain (kept by default, flagged), 23 drop.
- Downstream cross-check (several hundred papers of an earlier run with known outcome, no model labels): of the
  papers that became accepted tasks, rules-v1 keeps 85%, rules-v2 96%, Jev 96% (without the tuned joint-probability
  rule: 99.7%). Jev keeps 97% of those that failed later: S2 cannot predict downstream generation failures, and is
  not meant to.

**Speed and cost.** One request per paper with 10 questions: latency p50 0.53 s, p95 0.62–0.70 s (650 calls, direct
connection). Mean 1,400 input tokens per paper, about $0.00006 per paper. The rules take about 0.2 ms per paper:
Jev is thousands of times slower per paper but still well under a second; with `concurrency = 4` a batch of 400 papers
takes under a minute.

**Stability.** 20 papers asked 3 times each: no decision changed; largest probability spread 0.06, mean 0.009.

## Where they disagree

- Small-area estimation (Fay–Herriot) papers: rules-v1 counted "Survey" in "Living Conditions Survey" as a review
  cue (−2) and counted estimating / estimation / estimator / estimators as four cues. rules-v2 and Jev both keep
  them; the `is_review` question's criteria say explicitly that sample-survey data are not a literature survey.
- Rules keep papers whose abstract is full of method words but which are not method papers (a research briefing
  about a tool, a quantum-annealer hardware study, a review comparing two data-assimilation systems); Jev drops or
  flags them.
- Rules drop method papers with short or plainly worded abstracts (vehicle-routing branch-and-price, image
  clustering, copy-number normalisation, circle detection); Jev keeps them.
- Jev's 2 test misses are low-confidence reference keeps that it dropped on the joint-probability rule (a long-read
  sequencing method and a phase-unwrapping benchmark).
- No abstract: both screens defer to S4 and judge the first 4000 characters of the full text. On 11 such papers the
  rules kept 10 (rules-v1) / 11 (rules-v2) and Jev 11; the reference kept 9.
- Black-box / interactive form hint (all 200): Jev flagged all 3 papers that the reference marked interactive
  (Bayesian optimisation, active learning) plus 2 derivative-free optimisation papers; the regex flagged none of
  the 3 and flagged 7 papers the reference did not mark interactive.

## Caveats

- Reference labels come from one model with a rubric; 40% are low confidence. They are not a human gold set.
- The test half has 110 papers; the precision gain over rules-v1 is within noise. The recall gain is not.
- One threshold (`joint_no` = 0.35) was tuned on dev. Everything else uses TypeSafe's suggested 0.7 / 0.3.
- Thresholds are tied to `jev-1.14.0`; re-run this evaluation before moving to another version.

## Reproduce

```sh
cd gen
# build the sampled set (needs the earlier pipeline's data; see the docstring for the variables)
uv run python eval/screen_jev/build_set.py --out OUT/set.jsonl --corpus-out OUT/corpus.jsonl
# reference labels (teacher endpoint; hard USD cap via the budget ledger)
uv run python eval/screen_jev/label_opus.py --set OUT/set.jsonl --out OUT/labels.jsonl --usd-cap 5
# Jev answers (TYPESAFE_API_KEY in the environment; hard USD cap), then the reports
uv run python eval/screen_jev/run_eval.py call --set OUT/set.jsonl --jev-cache OUT/jev.jsonl --repeats 3
uv run python eval/screen_jev/run_eval.py report --set OUT/set.jsonl --labels OUT/labels.jsonl \
    --jev-cache OUT/jev.jsonl --out OUT/report.json
uv run python eval/screen_jev/run_eval.py call --set OUT/corpus.jsonl --jev-cache OUT/jev_corpus.jsonl
uv run python eval/screen_jev/run_eval.py corpus --set OUT/corpus.jsonl --jev-cache OUT/jev_corpus.jsonl
```

Keep `OUT` outside the repository: the per-paper files are not published here.
