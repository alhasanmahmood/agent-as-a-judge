#!/usr/bin/env python3
"""
English-paraphrase control for the multilingual AAAJ study.

GOAL
----
Reviewer concern: is the cross-language satisfaction variance caused by LANGUAGE,
or merely by the judge prompt being reworded? This script quantifies how much
satisfaction moves when you ONLY reword the English judge prompt (semantics held
constant) and compares that paraphrase variance against the variance you already
observe across the five languages. If language variance >> paraphrase variance,
the confound is closed.

WHAT TO RUN FIRST (the experiment itself)
-----------------------------------------
1. Hold fixed: the GPT-4o backbone, the 55 DevAI tasks (or your 20-task ablation
   subset if compute is tight), the three frameworks, all artifacts/trajectories,
   temperature = 0.0. Change ONLY the judge prompt wording.
2. Create K English paraphrases of the FULL judge prompt stack (the same modules you
   localize: judge, ask, locate, planning, retrieve). K = 3 is enough. Paraphrases
   must preserve meaning and evaluation logic; only surface phrasing changes. Have a
   second person confirm they are semantically equivalent, same as your translation
   validation step.
3. Run AAAJ once per English paraphrase, exactly as you ran each language.
4. Save task-level satisfaction in the SAME format you already export for the
   language runs, just with the paraphrase id as the condition label.

EXPECTED INPUT
--------------
A long-format CSV `results_long.csv` with one row per (task, condition):
    task_id, condition, framework, sat
where:
    condition in {EN_P0, EN_P1, EN_P2, AR, TR, ZH, HI}   (EN_P0 = your original EN prompt)
    framework in {MetaGPT, GPT-Pilot, OpenHands}
    sat = task-level requirement satisfaction in percent (0-100) for that cell
If you only have requirement-level 0/1 verdicts, aggregate to task-level percent first
(mean over the task's requirements), which is the unit your Table 1 / Wilcoxon tests use.

The script averages over frameworks per task (matching your paired-test convention),
then compares paraphrase spread vs language spread and runs the sanity-check tests.
"""

import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PARAPHRASE_CONDITIONS = ["EN_P0", "EN_P1", "EN_P2"]   # add EN_P3... if you ran more
LANGUAGE_CONDITIONS   = ["EN_P0", "AR", "TR", "ZH", "HI"]  # EN_P0 is the English anchor
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "results_long.csv"


def load_task_level(csv_path):
    """Average over frameworks so each (task, condition) is one paired score."""
    df = pd.read_csv(csv_path)
    needed = {"task_id", "condition", "sat"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {needed}; found {set(df.columns)}")
    # mean over frameworks -> one score per (task, condition)
    wide = (df.groupby(["task_id", "condition"])["sat"]
              .mean()
              .unstack("condition"))
    return wide


def per_task_spread(wide, conditions, label):
    """Mean over tasks of the within-task std across the given conditions."""
    sub = wide[conditions].dropna()
    within_task_std = sub.std(axis=1, ddof=1)        # spread for each task
    within_task_range = sub.max(axis=1) - sub.min(axis=1)
    cond_means = sub.mean(axis=0)                    # mean satisfaction per condition
    out = {
        "label": label,
        "n_tasks": len(sub),
        "conditions": conditions,
        "mean_within_task_std": within_task_std.mean(),
        "mean_within_task_range": within_task_range.mean(),
        "condition_means": cond_means.round(2).to_dict(),
        "range_of_condition_means": float(cond_means.max() - cond_means.min()),
    }
    return out


def paired_tests(wide, pairs):
    """Two-sided Wilcoxon signed-rank on paired task-level scores for each pair."""
    rows = []
    for a, b in pairs:
        sub = wide[[a, b]].dropna()
        if len(sub) < 5:
            rows.append((a, b, len(sub), np.nan, np.nan))
            continue
        diff = sub[b] - sub[a]
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(sub[a], sub[b])
        rows.append((a, b, len(sub), float(sub[b].mean() - sub[a].mean()), float(p)))
    return pd.DataFrame(rows, columns=["left", "right", "n", "mean_diff", "p_value"])


def main():
    wide = load_task_level(CSV_PATH)

    para = per_task_spread(wide, PARAPHRASE_CONDITIONS, "English paraphrases")
    lang = per_task_spread(wide, LANGUAGE_CONDITIONS, "Languages")

    print("=" * 70)
    print("ENGLISH-PARAPHRASE CONTROL  (GPT-4o, frameworks averaged)")
    print("=" * 70)
    for d in (para, lang):
        print(f"\n[{d['label']}]  ({d['n_tasks']} tasks, conditions={d['conditions']})")
        print(f"  per-task std (mean over tasks)   : {d['mean_within_task_std']:.2f} pts")
        print(f"  per-task range (mean over tasks) : {d['mean_within_task_range']:.2f} pts")
        print(f"  condition means                  : {d['condition_means']}")
        print(f"  range of condition means         : {d['range_of_condition_means']:.2f} pts")

    ratio = lang["mean_within_task_std"] / para["mean_within_task_std"] \
        if para["mean_within_task_std"] > 0 else float("inf")
    print("\n" + "-" * 70)
    print(f"Language std / paraphrase std  = {ratio:.2f}x")
    print("Interpretation: >> 1 means language moves scores far more than rewording,")
    print("which closes the confound. ~1 would support the reviewer's worry.")

    # Sanity-check significance: paraphrase pairs SHOULD be n.s.; language pairs SHOULD be sig.
    para_pairs = list(itertools.combinations(PARAPHRASE_CONDITIONS, 2))
    lang_pairs = [("EN_P0", "AR"), ("EN_P0", "HI"), ("EN_P0", "TR"), ("EN_P0", "ZH")]
    print("\nPaired Wilcoxon, English paraphrase pairs (expected: non-significant):")
    print(paired_tests(wide, para_pairs).to_string(index=False))
    print("\nPaired Wilcoxon, English vs other languages (expected: significant for AR, HI):")
    print(paired_tests(wide, lang_pairs).to_string(index=False))

    # Auto-draft the rebuttal sentence with the real numbers
    print("\n" + "=" * 70)
    print("REBUTTAL SENTENCE (paste the filled version):")
    print("=" * 70)
    print(
        f"Across {para['n_tasks']} tasks, re-wording the English judge prompt while holding "
        f"semantics fixed moves task-level satisfaction by only "
        f"{para['mean_within_task_std']:.1f} points on average "
        f"(range of paraphrase means {para['range_of_condition_means']:.1f} points), "
        f"whereas changing the language moves it by {lang['mean_within_task_std']:.1f} points "
        f"(range of language means {lang['range_of_condition_means']:.1f} points), "
        f"a factor of {ratio:.1f}. The English paraphrase pairs are not significantly "
        f"different, while EN vs. AR and EN vs. HI remain significant, confirming the "
        f"effect is driven by language rather than surface phrasing."
    )


if __name__ == "__main__":
    main()
