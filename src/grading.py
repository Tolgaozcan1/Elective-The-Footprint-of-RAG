"""
Phase E.3 — partial-credit grading rubric helpers.

This module supplies:
  * the per-fact SIGNALS lookup table (78 entries, one per atomic fact in
    dataset/eval_questions_atomic.md);
  * the FACT_COUNT denominator per question;
  * a normaliser, a fact-presence detector, and a grade-from-percentage mapping;
  * a single-cell grader (`grade_cell`) and a full-corpus driver (`grade_all`).

Design intent
-------------
The SIGNALS table maps each atomic fact (Q-id, F-id) to two signal sets:
  - "strict": tokens the primary grader (Eze) requires for "fact present"
  - "lenient": tokens the secondary grader (Tolga) requires for "fact present"
A fact is "present" iff every token in the chosen set appears as a
case-insensitive substring of the candidate answer (after CO2/CO₂
normalisation). The rubric in §11 E.3 Step 2 then maps the percentage of
facts present to a 0-5 grade. The two graders disagree on borderline cells
because their signal sets differ; that disagreement is what Cohen's kappa
quantifies in `notebooks/07_grading.ipynb`.

Refusals — answers containing the exact substring "I don't have enough
information" — are auto-graded 0 by both graders per the locked refusal
policy in MASTER_PLAN.md §11 E.3 / handoff brief §7.

Authors: Eze Nnaemeka Uzoma (primary grader), Tolga Özcan (secondary grader).
Atomic-fact list compiled by Saheed Yakubu.
"""

from __future__ import annotations

import re

REFUSAL_MARKER = "I don't have enough information"


SIGNALS: dict[tuple[str, str], dict[str, list[str]]] = {
    # ----- Q01 (BLOOM, factoid) -----
    ("Q01", "F1"): {"strict": ["50.5", "tonnes"],          "lenient": ["50", "co"]},
    ("Q01", "F2"): {"strict": ["11.2", "embodied"],        "lenient": ["11", "embodied"]},
    ("Q01", "F3"): {"strict": ["24.69", "dynamic"],        "lenient": ["24", "dynamic"]},
    ("Q01", "F4"): {"strict": ["14.6", "idle"],            "lenient": ["14", "idle"]},

    # ----- Q02 (BLOOM, factoid) -----
    ("Q02", "F1"): {"strict": ["57", "french"],            "lenient": ["57", "france"]},
    ("Q02", "F2"): {"strict": ["jean zay"],                "lenient": ["jean zay"]},
    ("Q02", "F3"): {"strict": ["429", "gpt"],              "lenient": ["429"]},
    ("Q02", "F4"): {"strict": ["502", "gpt"],              "lenient": ["502"]},
    ("Q02", "F5"): {"strict": ["25", "tonnes"],            "lenient": ["25", "bloom"]},
    ("Q02", "F6"): {"strict": ["20 times"],                "lenient": ["20"]},
    ("Q02", "F7"): {"strict": ["433", "mwh"],              "lenient": ["433"]},
    ("Q02", "F8"): {"strict": ["324", "opt"],              "lenient": ["324"]},

    # ----- Q03 (BLOOM, synthesis) -----
    ("Q03", "F1"): {"strict": ["a100", "jean zay"],        "lenient": ["a100"]},
    ("Q03", "F2"): {"strict": ["54", "dynamic"],           "lenient": ["54"]},
    ("Q03", "F3"): {"strict": ["64", "idle"],              "lenient": ["64"]},
    ("Q03", "F4"): {"strict": ["27", "infrastructure"],    "lenient": ["27"]},
    ("Q03", "F5"): {"strict": ["pue"],                     "lenient": ["pue"]},
    ("Q03", "F6"): {"strict": ["dynamic", "understate"],   "lenient": ["understate"]},

    # ----- Q04 (IEA, numerical) -----
    ("Q04", "F1"): {"strict": ["415", "twh"],              "lenient": ["415"]},
    ("Q04", "F2"): {"strict": ["1.5%"],                    "lenient": ["1.5"]},
    ("Q04", "F3"): {"strict": ["945", "twh"],              "lenient": ["945"]},
    ("Q04", "F4"): {"strict": ["japan"],                   "lenient": ["japan"]},

    # ----- Q05 (IEA, factoid) -----
    ("Q05", "F1"): {"strict": ["16 trillion"],             "lenient": ["16"]},
    ("Q05", "F2"): {"strict": ["12 trillion"],             "lenient": ["12"]},
    ("Q05", "F3"): {"strict": ["75%"],                     "lenient": ["75"]},
    ("Q05", "F4"): {"strict": ["chatgpt"],                 "lenient": ["generative ai"]},

    # ----- Q06 (IEA, synthesis) -----
    ("Q06", "F1"): {"strict": ["20%", "delay"],            "lenient": ["20"]},
    ("Q06", "F2"): {"strict": ["four to eight", "transmission"], "lenient": ["transmission"]},
    ("Q06", "F3"): {"strict": ["doubled", "transformer"],  "lenient": ["transformer"]},
    ("Q06", "F4"): {"strict": ["locating", "data centres"], "lenient": ["locating"]},
    ("Q06", "F5"): {"strict": ["onsite", "backup"],        "lenient": ["backup"]},
    ("Q06", "F6"): {"strict": ["175 gw"],                  "lenient": ["175"]},

    # ----- Q07 (Google, factoid) -----
    ("Q07", "F1"): {"strict": ["12%", "emissions"],        "lenient": ["12"]},
    ("Q07", "F2"): {"strict": ["27%", "consumption"],      "lenient": ["27"]},
    ("Q07", "F3"): {"strict": ["clean energy"],            "lenient": ["clean"]},
    ("Q07", "F4"): {"strict": ["25", "projects"],          "lenient": ["projects"]},
    ("Q07", "F5"): {"strict": ["2.5 gw"],                  "lenient": ["2.5"]},
    ("Q07", "F6"): {"strict": ["hardware", "efficiency"],  "lenient": ["efficiency"]},

    # ----- Q08 (Google, factoid) -----
    ("Q08", "F1"): {"strict": ["ironwood", "seventh"],     "lenient": ["ironwood"]},
    ("Q08", "F2"): {"strict": ["30 times", "first"],       "lenient": ["30"]},
    ("Q08", "F3"): {"strict": ["double", "trillium"],      "lenient": ["trillium"]},
    ("Q08", "F4"): {"strict": ["threefold", "tpu v4"],     "lenient": ["threefold"]},
    ("Q08", "F5"): {"strict": ["fewer", "carbon"],         "lenient": ["fewer carbon"]},

    # ----- Q09 (Google, synthesis) -----
    ("Q09", "F1"): {"strict": ["12%", "emissions"],        "lenient": ["12"]},
    ("Q09", "F2"): {"strict": ["27%", "consumption"],      "lenient": ["27"]},
    ("Q09", "F3"): {"strict": ["scope 3"],                 "lenient": ["supply chain"]},
    ("Q09", "F4"): {"strict": ["carbon-free", "deployment"], "lenient": ["carbon-free"]},
    ("Q09", "F5"): {"strict": ["ai", "demand"],            "lenient": ["demand"]},
    ("Q09", "F6"): {"strict": ["policy"],                  "lenient": ["policy"]},
    ("Q09", "F7"): {"strict": ["asia", "pacific"],         "lenient": ["asia"]},
    ("Q09", "F8"): {"strict": ["suppliers"],               "lenient": ["suppliers"]},

    # ----- Q10 (EPRI, numerical) -----
    ("Q10", "F1"): {"strict": ["2.9", "wh"],               "lenient": ["2.9"]},
    ("Q10", "F2"): {"strict": ["0.3", "wh"],               "lenient": ["0.3"]},
    ("Q10", "F3"): {"strict": ["10 times"],                "lenient": ["10"]},
    ("Q10", "F4"): {"strict": ["80", "gwh"],               "lenient": ["80"]},
    ("Q10", "F5"): {"strict": ["29.2", "twh"],             "lenient": ["29.2"]},
    ("Q10", "F6"): {"strict": ["400,000", "servers"],      "lenient": ["400"]},
    ("Q10", "F7"): {"strict": ["62.4"],                    "lenient": ["62"]},

    # ----- Q11 (EPRI, factoid) -----
    ("Q11", "F1"): {"strict": ["development", "training", "inference"], "lenient": ["inference"]},
    ("Q11", "F2"): {"strict": ["10%", "development"],      "lenient": ["10"]},
    ("Q11", "F3"): {"strict": ["30%", "training"],         "lenient": ["30"]},
    ("Q11", "F4"): {"strict": ["60%", "inference"],        "lenient": ["60"]},

    # ----- Q12 (Greenpeace, numerical) -----
    ("Q12", "F1"): {"strict": ["351%"],                    "lenient": ["350"]},
    ("Q12", "F2"): {"strict": ["218", "gwh"],              "lenient": ["218"]},
    ("Q12", "F3"): {"strict": ["984", "gwh"],              "lenient": ["984"]},
    ("Q12", "F4"): {"strict": ["37,238"],                  "lenient": ["37"]},
    ("Q12", "F5"): {"strict": ["170-fold"],                "lenient": ["170"]},
    ("Q12", "F6"): {"strict": ["ireland"],                 "lenient": ["ireland"]},

    # ----- Q13 (BLOOM + Greenpeace, multi-doc) -----
    ("Q13", "F1"): {"strict": ["embodied", "22.2%"],       "lenient": ["embodied"]},
    ("Q13", "F2"): {"strict": ["idle", "28.9%"],           "lenient": ["idle"]},
    ("Q13", "F3"): {"strict": ["embodied", "idle", "omitted"], "lenient": ["omitted"]},
    ("Q13", "F4"): {"strict": ["dynamic", "gpu"],          "lenient": ["gpu"]},
    ("Q13", "F5"): {"strict": ["dynamic", "understate"],   "lenient": ["understate"]},
    ("Q13", "F6"): {"strict": ["chip", "manufacturing"],   "lenient": ["chipmaking"]},
    ("Q13", "F7"): {"strict": ["taiwan"],                  "lenient": ["taiwan"]},
    ("Q13", "F8"): {"strict": ["350%"],                    "lenient": ["350"]},
    ("Q13", "F9"): {"strict": ["453,600"],                 "lenient": ["453"]},
    ("Q13", "F10"): {"strict": ["lifecycle"],              "lenient": ["lifecycle"]},
}


FACT_COUNT: dict[str, int] = {
    "Q01": 4, "Q02": 8, "Q03": 6, "Q04": 4, "Q05": 4,
    "Q06": 6, "Q07": 6, "Q08": 5, "Q09": 8, "Q10": 7,
    "Q11": 4, "Q12": 6, "Q13": 10,
}

assert sum(FACT_COUNT.values()) == 78, "Total atomic facts must equal 78."
assert len(SIGNALS) == 78, f"SIGNALS must have 78 entries, got {len(SIGNALS)}."


def normalise(text: str) -> str:
    """Lowercase, harmonise CO2/CO₂ variants, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.replace("co₂", "co2")
    t = re.sub(r"\s+", " ", t)
    return t


def fact_present(answer_norm: str, signals: list[str]) -> bool:
    """A fact is present iff every signal token appears as a substring."""
    return all(sig.lower() in answer_norm for sig in signals)


def grade_for_pct(pct: float, has_contradiction: bool = False) -> int:
    """Map fact-presence percentage to 0-5 grade per MASTER_PLAN §11 E.3 Step 2."""
    if has_contradiction:
        return 1
    if pct >= 1.0:
        return 5
    if pct >= 0.75:
        return 4
    if pct >= 0.5:
        return 3
    if pct >= 0.25:
        return 2
    return 1


def grade_cell(answer_text: str, question_id: str, mode: str) -> tuple[int, list[str]]:
    """
    Grade one (question, candidate-answer) pair under the chosen signal set.

    Args:
        answer_text: candidate answer string from query_log.csv
        question_id: e.g. "Q01" .. "Q13"
        mode: "strict" (Eze) or "lenient" (Tolga)

    Returns:
        (grade, list of present-fact ids)
    """
    answer_norm = normalise(answer_text)
    n_facts = FACT_COUNT[question_id]
    present: list[str] = []

    for fact_idx in range(1, n_facts + 1):
        fact_id = f"F{fact_idx}"
        signals = SIGNALS[(question_id, fact_id)][mode]
        if fact_present(answer_norm, signals):
            present.append(fact_id)

    pct = len(present) / n_facts
    grade = grade_for_pct(pct, has_contradiction=False)
    return grade, present


def is_refusal(answer_text: str) -> bool:
    """The §8 baseline prompt instructs the LLM to emit REFUSAL_MARKER on failure."""
    if not isinstance(answer_text, str):
        return False
    return REFUSAL_MARKER in answer_text
