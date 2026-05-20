# Atomic Facts for Reference Answers

**Phase:** E.3 Step 1 — atomic-fact decomposition for the partial-credit grading rubric (MASTER_PLAN.md §11 E.3)
**Prepared by:** Saheed Yakubu
**Date:** 2026-05-10
**Companion files:** `dataset/eval_questions.md` (the 13 questions and their reference answers), `dataset/pdfs/` (source PDFs used for cross-checking).

## Method

For each of the 13 evaluation questions, the reference answer in `eval_questions.md` was read in full and decomposed into discrete, independently verifiable atomic facts. A "fact" here is a single verifiable statement: one number paired with one label, one cause paired with one effect, or one named entity with one attribute. Rhetorical framing sentences ("This reflects…", "This was made possible by…") were dropped when they did not carry verifiable claims; where such a sentence anchored a specific name, percentage, or concrete driver, that anchor was kept as its own fact.

Numerical facts retain the precision used in the reference answer — no precision was invented. Where the reference answer hedges with "approximately" or "roughly", the same hedge is preserved in the atomic fact so that a candidate answer matching the hedge is not penalised for not exceeding it.

For Question 13 (multi-doc, BLOOM + Greenpeace), facts are numbered sequentially `F1, F2, …` across both source documents per the locked formatting decision (Phase E / Batch B). Each Q13 fact carries a `(source: …)` tag with the page range from the question's source-location field. For Q01–Q12, the source document is implicit in the question header and source tags are not used.

Numerical claims in this file were independently cross-checked against the cited PDF pages in `dataset/pdfs/`. No discrepancies were found between the reference-answer numbers and the source PDFs at the time of finalisation.

## Notes for graders

- Each fact contributes equally; no per-fact weighting.
- The 0–5 grade mapping is count-based and lives in MASTER_PLAN.md §11 E.3 Step 2.
- A candidate answer that gives the right *label* but the wrong *number* is a contradiction — caps the grade at 1 per the rubric, regardless of how many other facts are present. Flag such cells in the grading-log `notes` column.
- For Q13, the strict-vs-lenient retrieval framing (Phase E Batch A, `recall_log.csv`) is a separate diagnostic. Here, treat each tagged fact independently — a candidate answer covering only one source document well can still earn ≥3.

---

## Question 01

**Reference answer atomic facts:**

- F1: Total CO₂eq is approximately 50.5 tonnes
- F2: Embodied emissions are 11.2 tonnes (22.2%)
- F3: Dynamic power emissions are 24.69 tonnes (48.9%)
- F4: Idle power emissions are 14.6 tonnes (28.9%)

---

## Question 02

**Reference answer atomic facts:**

- F1: The French electricity grid powering Jean Zay has a carbon intensity of 57 gCO₂eq/kWh
- F2: BLOOM was trained on the Jean Zay cluster at IDRIS
- F3: GPT-3 was trained on a grid with a carbon intensity of 429 gCO₂eq/kWh
- F4: GPT-3 training emitted approximately 502 tonnes of CO₂eq
- F5: BLOOM training emitted approximately 25 tonnes of CO₂eq
- F6: GPT-3's emissions were roughly 20 times higher than BLOOM's
- F7: BLOOM consumed 433 MWh of energy during training
- F8: OPT consumed 324 MWh of energy during training

---

## Question 03

**Reference answer atomic facts:**

- F1: The authors ran direct measurements on the Jean Zay A100 partition to capture infrastructure, idle, and dynamic consumption
- F2: Approximately 54% of total power was attributable to the dynamic GPU workload (109 kW out of 200 kW)
- F3: Idle servers accounted for 64 kW of power
- F4: Always-on infrastructure accounted for 27 kW of power
- F5: PUE-based accounting omits the idle consumption of servers that are powered on but not actively computing
- F6: Exclusive focus on dynamic GPU consumption understates actual training emissions by nearly half

---

## Question 04

**Reference answer atomic facts:**

- F1: Global data centres consumed approximately 415 TWh of electricity in 2024
- F2: Data centres accounted for approximately 1.5% of world electricity consumption in 2024
- F3: The IEA Base Case projects data centre consumption to reach approximately 945 TWh by 2030
- F4: The projected 2030 figure is roughly equivalent to Japan's current total electricity consumption

---

## Question 05

**Reference answer atomic facts:**

- F1: S&P 500 market capitalisation increased by USD 16 trillion since November 2022
- F2: Of that increase, USD 12 trillion came from AI-related companies
- F3: AI-related companies accounted for approximately 75% of the S&P 500 market-cap growth
- F4: The investor surge is attributed to the commercial launch of generative AI products such as ChatGPT

---

## Question 06

**Reference answer atomic facts:**

- F1: Around 20% of planned data centre projects are at risk of delays due to grid connection constraints
- F2: Building new transmission lines takes four to eight years in advanced economies
- F3: Wait times for transformers and cables have doubled in the past three years
- F4: Locating data centres in areas of high power and grid availability is a mitigation strategy
- F5: Operating server capacity or onsite backup power and storage more flexibly is a mitigation strategy
- F6: AI-based grid management tools could unlock up to 175 GW of transmission capacity through remote sensors

---

## Question 07

**Reference answer atomic facts:**

- F1: Google reduced its data centre energy emissions by 12% in 2024 compared to 2023
- F2: Google's data centre electricity consumption grew by 27% year-on-year in 2024
- F3: Clean energy procurement was the primary enabler of the emissions reduction
- F4: More than 25 clean energy projects contracted in prior years came online in 2024
- F5: 2.5 GW of new clean energy was added to grids serving Google's operations in 2024
- F6: Ongoing improvements in hardware efficiency also contributed to the reduction

---

## Question 08

**Reference answer atomic facts:**

- F1: Ironwood is Google's seventh-generation TPU, announced in April 2025
- F2: Ironwood is nearly 30 times more power-efficient than the first Cloud TPU (2018)
- F3: Ironwood's performance per watt is double that of the Trillium (sixth-generation) TPU
- F4: A 2025 Google study found a threefold improvement in Compute Carbon Intensity from TPU v4 to Trillium over four years
- F5: Newer TPU generations deliver cutting-edge AI performance while generating fewer carbon emissions per equivalent workload

---

## Question 09

**Reference answer atomic facts:**

- F1: Google reduced data centre energy emissions by 12% in 2024
- F2: Google's total electricity consumption grew by 27% in 2024
- F3: Supply chain (Scope 3) emissions rose even as operational emissions fell
- F4: Slower-than-needed deployment of carbon-free energy technologies is cited as a driver of the divergence
- F5: The growing energy demands of AI are cited as a driver of the divergence
- F6: Policy uncertainties are cited as a driver of the divergence
- F7: Resource-challenged markets, particularly the Asia-Pacific region with less accessible wind and solar resources, are cited as a driver
- F8: Difficulty compelling suppliers to match 100% clean electricity for product manufacturing is cited as a driver

---

## Question 10

**Reference answer atomic facts:**

- F1: A ChatGPT query consumes approximately 2.9 Wh of electricity
- F2: A traditional Google search consumes approximately 0.3 Wh of electricity
- F3: ChatGPT queries consume roughly 10 times the electricity of a Google search
- F4: SemiAnalysis estimates AI-integrated Google searches would require approximately 80 GWh daily
- F5: SemiAnalysis estimates AI-integrated Google searches would require 29.2 TWh annually
- F6: New Street Research estimates AI integration would require around 400,000 additional servers
- F7: New Street Research estimates AI integration would require 62.4 GWh daily / 22.8 TWh yearly

---

## Question 11

**Reference answer atomic facts:**

- F1: EPRI categorises AI annual energy workloads into three phases: model development, model training, and use/inference
- F2: Model development accounts for approximately 10% of the AI energy footprint
- F3: Model training accounts for approximately 30% of the AI energy footprint
- F4: Use/inference accounts for approximately 60% of the AI energy footprint

---

## Question 12

**Reference answer atomic facts:**

- F1: Global electricity consumption from AI chip manufacturing increased by approximately 351% between 2023 and 2024
- F2: AI chipmaking electricity consumption was 218 GWh in 2023
- F3: AI chipmaking electricity consumption was approximately 984 GWh in 2024
- F4: Greenpeace's most ambitious growth scenario projects AI chipmaking consumption could reach up to 37,238 GWh by 2030
- F5: The 2030 projection represents a 170-fold increase compared to 2023
- F6: The projected 2030 figure would exceed Ireland's current total electricity consumption

---

## Question 13

**Reference answer atomic facts:**

- F1: Embodied emissions from manufacturing computing equipment add 22.2% to BLOOM's training footprint (source: BLOOM, pp.4-7)
- F2: Idle power consumption — the energy used by servers that are powered on but not actively training — adds 28.9% to BLOOM's training footprint (source: BLOOM, pp.4-7)
- F3: Embodied emissions and idle power are typically omitted from standard AI carbon footprint estimates (source: BLOOM, pp.4-7)
- F4: Most AI carbon research focuses only on GPU dynamic power consumption (source: BLOOM, pp.4-7)
- F5: Dynamic-only accounting understates BLOOM's training footprint by nearly half (source: BLOOM, pp.4-7)
- F6: The upstream electricity consumption of AI chip manufacturing is almost entirely absent from current AI climate assessments (source: Greenpeace, pp.10-11)
- F7: AI chip manufacturing is concentrated in fossil-fuel-heavy East Asian grids (Taiwan, South Korea, Japan) (source: Greenpeace, pp.10-11)
- F8: AI chipmaking electricity consumption grew by more than 350% in a single year (source: Greenpeace, pp.16-22)
- F9: AI chipmaking generated over 453,600 metric tons of CO₂eq in 2024 (source: Greenpeace, pp.16-22)
- F10: Both reports conclude that a complete, lifecycle-based accounting framework is necessary to represent the full climate impact of AI infrastructure (source: BLOOM + Greenpeace)

---

_End of eval_questions_atomic.md_
