"""Funnel cost model for DESIGN.md §4 (estimates only; parameters are assumptions, not measurements).

Run: python3 cost_model.py
Unit: per 100 retrieved candidate papers. "suitable" = passed all gates AND GLM-5.3 is not stable on it
(first run fails and a teacher passes, or first run passes and second fails), or a harder variant that lands there.
Prices are placeholders: Opus-class $5/M input, $25/M output; cheap model $0.5/M input, $2/M output.
GLM / teacher trial dollars use Harbor-recorded cost_usd (LiteLLM price map), not real invoices.
"""

BASE = dict(
    screen=0.5,      # fraction of retrieved candidates kept by rule+small-model screening   (assumption)
    ft=0.85,         # full text obtained                                                      (assumption; download/REPORT.md)
    feas=0.6,        # feasibility probe passes (data/code runnable, not duplicate)           (assumption)
    spec_ok=0.9,     # spec passes mechanical spec checks within 1 retry                       (assumption)
    gate_ok=0.8,     # passes all mechanical gates after <=2 local repairs                     (old: 244/309=0.79)
    f1=0.4,          # GLM run-1 fail rate on gated tasks                                     (old: 63/238=0.26; target harder)
    p2=0.25,         # GLM run-2 fail rate given run-1 pass                                    (old: 29/175=0.17)
    t_ok=0.6,        # teacher passes when GLM run-1 failed                                    (old GLM-0/3 tasks: 23/26)
    t_runs=1.3,      # teacher runs per GLM-failed task
    var_try=0.7,     # fraction of easy tasks sent to harder-variant generation
    var_gate=0.7,    # variant passes gates
    spec_in=40e3, spec_out=10e3,
    gA_in=30e3, gA_out=25e3,   # stage A: generator + reference solution
    gB_in=45e3, gB_out=15e3,   # stage B: baselines + tests + instruction
    rep_n=0.8, rep_in=35e3, rep_out=6e3,   # patch-style repair rounds per generated task
    var_in=45e3, var_out=15e3,
    feas_in=400e3, feas_out=12e3,          # cheap-model agent probe per full text
)
OPT = dict(BASE, feas=0.75, gate_ok=0.85, f1=0.5, p2=0.3, t_ok=0.7, t_runs=1.2, rep_n=0.5, feas_in=250e3)
PESS = dict(BASE, feas=0.4, gate_ok=0.65, f1=0.27, p2=0.17, t_ok=0.5, t_runs=2, rep_n=1.2, feas_in=800e3,
            spec_out=15e3, gA_out=35e3, gB_out=20e3, var_gate=0.5)


def run(p, N=100, glm_usd=11.2, teacher_usd=6.0):
    ft = N * p["screen"] * p["ft"]
    feas = ft * p["feas"]
    gen = feas * p["spec_ok"]
    gates = gen * p["gate_ok"]
    f1, p2 = p["f1"], p["p2"]
    easy = gates * (1 - f1) * (1 - p2)
    unstable = gates * (1 - f1) * p2
    hard = gates * f1
    var_n = easy * p["var_try"]
    var_gates = var_n * p["var_gate"]
    var_suit = var_gates * (1 - (1 - f1) * (1 - p2))
    suit = unstable + hard * p["t_ok"] + var_suit
    oi = feas * p["spec_in"] + gen * (p["gA_in"] + p["gB_in"] + p["rep_n"] * p["rep_in"]) + var_n * (p["var_in"] + p["rep_n"] * p["rep_in"])
    oo = feas * p["spec_out"] + gen * (p["gA_out"] + p["gB_out"] + p["rep_n"] * p["rep_out"]) + var_n * (p["var_out"] + p["rep_n"] * p["rep_out"])
    ci = N * 3e3 + ft * p["feas_in"] + (gen + var_n) * (30e3 + p["rep_n"] * 15e3) + (gates + var_gates) * 20e3
    co = N * 0.3e3 + ft * p["feas_out"] + (gen + var_n) * (3e3 + p["rep_n"] * 1.5e3) + (gates + var_gates) * 2e3
    glm = (gates + var_gates) * (1 + (1 - f1))
    teach = (gates + var_gates) * f1 * p["t_runs"]
    opus_usd = (oi * 5 + oo * 25) / 1e6
    cheap_usd = (ci * 0.5 + co * 2) / 1e6
    return dict(fulltext=ft, feasible=feas, generated=gen, gated=gates, easy=easy, unstable=unstable, hard=hard,
                variants_gated=var_gates, suitable=suit,
                opus_out_per=oo / suit, opus_in_per=oi / suit, opus_usd_per=opus_usd / suit,
                cheap_in_per=ci / suit, cheap_usd_per=cheap_usd / suit,
                glm_trials_per=glm / suit, teacher_trials_per=teach / suit,
                total_usd_per=(opus_usd + cheap_usd + glm * glm_usd + teach * teacher_usd) / suit)


if __name__ == "__main__":
    for name, p in [("optimistic", OPT), ("base", BASE), ("pessimistic", PESS)]:
        for g in (11.2, 30.0):
            r = run(p, glm_usd=g)
            print(name, f"glm_trial=${g}", {k: (round(v, 2) if abs(v) < 1000 else int(v)) for k, v in r.items()})
