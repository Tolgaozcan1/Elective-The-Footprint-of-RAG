# Evaluation Questions

Total: 13 questions across 5 documents. Questions are stratified per document to balance
evaluation against an intentionally heterogeneous corpus. All questions are answerable
from narrative text extracted by pypdf (not from tables, charts, or images).

## Distribution

| Document | Question count |
|----------|---------------:|
| BLOOM | 3 |
| IEA | 3 |
| Google | 3 |
| EPRI | 2 |
| Greenpeace | 2 |
| **Total** | **13** |

| Question type | Count |
|---------------|------:|
| factoid | 6 |
| numerical | 3 |
| synthesis | 3 |
| multi-doc | 1 |

---

## Question 01

**Source document:** 01_BLOOM_Luccioni_2022.pdf
**Question type:** factoid
**Question:** What was the total carbon footprint of training the BLOOM 176B model when
accounting for all lifecycle processes — embodied emissions, dynamic power consumption,
and idle consumption — and how does each component contribute?
**Reference answer:** The total carbon footprint of BLOOM's final training run was
approximately 50.5 tonnes of CO₂eq. This breaks down into 11.2 tonnes from embodied
(manufacturing) emissions (22.2%), 24.69 tonnes from dynamic power consumption (48.9%),
and 14.6 tonnes from idle power consumption (28.9%).
**Source location:** Section 4 (Results), Table 3 (page 7)

---

## Question 02

**Source document:** 01_BLOOM_Luccioni_2022.pdf
**Question type:** factoid
**Question:** Why did BLOOM's training produce far fewer CO₂ emissions than comparable
large language models such as GPT-3, even though it consumed slightly more energy than OPT?
**Reference answer:** BLOOM's relatively low emissions were primarily due to the low carbon
intensity of the French electricity grid (57 gCO₂eq/kWh) powering the Jean Zay computing
cluster at IDRIS. By contrast, GPT-3 was trained on a grid with a carbon intensity of
429 gCO₂eq/kWh, which resulted in roughly 502 tonnes of CO₂eq — approximately 20 times
more than BLOOM's 25 tonnes — despite BLOOM consuming more raw energy (433 MWh)
than OPT (324 MWh).
**Source location:** Section 5.1 (Comparisons with other LLMs), pages 9–10

---

## Question 03

**Source document:** 01_BLOOM_Luccioni_2022.pdf
**Question type:** synthesis
**Question:** How did the authors measure idle power consumption during BLOOM training,
and why do they argue that relying solely on GPU dynamic consumption (or datacenter PUE)
understates the true emissions of model training?
**Reference answer:** The authors ran experiments on the Jean Zay A100 partition to directly
measure infrastructure, idle, and dynamic consumption, finding that only about 54% of total
power (109 kW out of 200 kW) was attributable to the dynamic GPU workload; the remaining
46% covered idle servers (64 kW) and always-on infrastructure (27 kW). They argue that
PUE-based accounting does not capture this overhead because PUE only divides total
facility energy by IT equipment energy, omitting the idle consumption of servers that are
powered on but not actively computing. As a result, exclusive focus on dynamic GPU
consumption understates actual emissions by nearly half.
**Source location:** Section 4.3 (Idle Power Consumption), Table 2, pages 6–7

---

## Question 04

**Source document:** 02_IEA_Energy_and_AI_2025.pdf
**Question type:** numerical
**Question:** How much electricity did global data centres consume in 2024, and what does
the IEA project their consumption will reach by 2030 in its Base Case?
**Reference answer:** Global data centres consumed approximately 415 terawatt-hours (TWh)
in 2024, accounting for around 1.5% of world electricity consumption. In the IEA's Base
Case, this figure is projected to more than double to around 945 TWh by 2030 — roughly
equivalent to Japan's total electricity consumption today.
**Source location:** Executive Summary, pages 13–14

---

## Question 05

**Source document:** 02_IEA_Energy_and_AI_2025.pdf
**Question type:** factoid
**Question:** According to the IEA, how much of the USD 16 trillion growth in S&P 500 market
capitalisation since 2022 came from AI-related companies, and what does this indicate about
the pace of AI industry growth?
**Reference answer:** Of the USD 16 trillion increase in market capitalisation of S&P 500
companies since November 2022, USD 12 trillion — approximately 75% — came from
AI-related companies. This reflects a surge in investor expectations driven by the commercial
launch of generative AI products such as ChatGPT and rapid adoption across corporate
strategies, economic policies, and geopolitics.
**Source location:** Section 1.2 (The rise of AI) / Section 1.2.1, pages 21–22

---

## Question 06

**Source document:** 02_IEA_Energy_and_AI_2025.pdf
**Question type:** synthesis
**Question:** The IEA identifies grid infrastructure as a significant bottleneck for data centre
expansion. What is the estimated scale of this risk, and what strategies does the report
suggest to mitigate it without necessarily building new transmission lines?
**Reference answer:** The IEA estimates that around 20% of planned data centre projects
could be at risk of delays due to grid connection constraints, as building new transmission
lines can take four to eight years in advanced economies and wait times for critical
components such as transformers and cables have doubled in the past three years. To
mitigate these risks without new lines, the report points to locating data centres in areas of
high power and grid availability, operating server capacity or onsite backup power and
storage more flexibly, and applying AI-based grid management tools — noting that up to
175 GW of transmission capacity could be unlocked through remote sensors and AI-based
management, exceeding the additional data centre power load projected to 2030.
**Source location:** Executive Summary, pages 14–16

---

## Question 07

**Source document:** 03_Google_Env_2025.pdf
**Question type:** factoid
**Question:** In 2024, Google reduced its data centre energy emissions despite significantly
growing its electricity consumption. What were the specific percentage changes in each
metric, and what is cited as the primary reason this was possible?
**Reference answer:** In 2024, Google reduced its data centre energy emissions by 12%
compared to 2023, despite its data centre electricity consumption growing by 27% year-on-year.
This was made possible by clean energy procurement — including more than 25 projects
contracted over prior years that came online in 2024, adding 2.5 GW of new clean energy
to the grids that serve Google's operations — combined with ongoing improvements in
hardware efficiency.
**Source location:** Section "Scaling smarter: Successfully reducing our data center energy
emissions," pages 19–20 (also highlighted in the Foreword, page 4)

---

## Question 08

**Source document:** 03_Google_Env_2025.pdf
**Question type:** factoid
**Question:** How much more power-efficient is Google's Ironwood TPU compared to its
first Cloud TPU from 2018, and what broader trend does this represent in Google's TPU
hardware over four years?
**Reference answer:** Google's Ironwood (seventh-generation TPU, announced April 2025) is
nearly 30 times more power-efficient than the company's first Cloud TPU from 2018, with
performance per watt double that of the Trillium (sixth-generation) TPU. More broadly,
a 2025 Google study found a threefold improvement in Compute Carbon Intensity of its TPU
chips over four years, from TPU v4 to Trillium, meaning newer generations deliver
cutting-edge AI performance while generating fewer carbon emissions per equivalent workload.
**Source location:** Section "AI efficiency gains: Improvements across models, TPUs, and
infrastructure" / "2024 highlights," pages 14–15 and page 8

---

## Question 09

**Source document:** 03_Google_Env_2025.pdf
**Question type:** synthesis
**Question:** Google's 2025 Environmental Report acknowledges an apparent contradiction
in its environmental progress: operational emissions fell while overall climate ambitions
became harder to achieve. What factors does the report identify as driving this divergence?
**Reference answer:** Google reduced its data centre energy emissions by 12% while
simultaneously increasing its total electricity consumption by 27%, demonstrating progress
on operational decarbonisation. However, the report acknowledges that supply chain
(Scope 3) emissions rose even as operational emissions fell. The divergence is attributed
to several external factors largely outside Google's direct control: slower-than-needed
deployment of carbon-free energy technologies, the growing energy demands of AI,
policy uncertainties, resource-challenged markets (particularly the Asia-Pacific region where
wind and solar resources are less accessible), and the difficulty of compelling suppliers
to match 100% clean electricity for Google product manufacturing.
**Source location:** Section "Ambitious vision, complex reality: Our efforts to decarbonize
global grids" and Foreword, pages 5 and 29

---

## Question 10

**Source document:** 04_EPRI_Powering_Intelligence_2024.pdf
**Question type:** numerical
**Question:** How does the electricity consumption of a typical ChatGPT query compare to
a traditional Google search, according to EPRI, and what are the implications if AI were
integrated into all Google searches?
**Reference answer:** EPRI estimates that a ChatGPT query consumes approximately
2.9 watt-hours (Wh) of electricity, roughly 10 times the 0.3 Wh consumed by a traditional
Google search. If AI were integrated into every Google search, one analysis cited by EPRI
(SemiAnalysis) estimates this could necessitate approximately 80 gigawatt-hours (GWh)
daily or 29.2 terawatt-hours (TWh) annually; another analysis (New Street Research)
estimates around 400,000 additional servers consuming 62.4 GWh daily or 22.8 TWh yearly.
**Source location:** Section "Chat GPT and Other Large Language Models (LLMs)," pages 15–16

---

## Question 11

**Source document:** 04_EPRI_Powering_Intelligence_2024.pdf
**Question type:** factoid
**Question:** How are AI annual energy workloads categorised by EPRI, and what share of
total AI energy does each phase account for?
**Reference answer:** EPRI categorises AI annual energy workloads into three phases:
model development (approximately 10% of the energy footprint), model training
(approximately 30% of the energy footprint), and use/inference (approximately 60% of
the energy footprint). The inference phase — the deployment and utilisation of trained
models in real-world applications — therefore accounts for the largest single share of
AI-related energy consumption.
**Source location:** Section "AI Implications for Power Consumption," page 15

---

## Question 12

**Source document:** 05_Greenpeace_Chipping_Point_2025.pdf
**Question type:** numerical
**Question:** By how much did global electricity consumption from AI chip manufacturing
increase between 2023 and 2024, and what does Greenpeace project this figure could reach
by 2030?
**Reference answer:** Global electricity consumption from AI chip manufacturing increased
by approximately 351%, from 218 GWh in 2023 to approximately 984 GWh in 2024. Under the
most ambitious growth scenario, Greenpeace projects this figure could reach up to
37,238 GWh by 2030 — a 170-fold increase compared to 2023 — which would exceed
Ireland's current total electricity consumption.
**Source location:** Section "Energy demand" / "Key Findings," pages 7 and 20

---

## Question 13

**Source document:** 01_BLOOM_Luccioni_2022.pdf + 05_Greenpeace_Chipping_Point_2025.pdf
**Question type:** multi-doc
**Question:** Both the BLOOM paper and the Greenpeace Chipping Point report identify
categories of AI hardware emissions that are typically overlooked in standard carbon
accounting. What are these overlooked categories, and why does each report argue they
should be included in a complete emissions assessment?
**Reference answer:** The BLOOM paper argues that embodied emissions (from manufacturing
computing equipment) and idle power consumption (the energy used by servers that are
powered on but not actively training) are routinely omitted from AI carbon footprint
estimates because most research focuses only on GPU dynamic power consumption. The
authors show these two factors add 22.2% and 28.9% respectively to BLOOM's total
training emissions, meaning dynamic-only accounting understates the footprint by nearly
half. The Greenpeace Chipping Point report argues that the upstream electricity consumption
of AI chip manufacturing — concentrated in fossil-fuel-heavy East Asian grids (Taiwan,
South Korea, Japan) — is almost entirely absent from current AI climate assessments,
despite growing by more than 350% in a single year and generating over 453,600 metric
tons of CO₂eq in 2024. Both reports conclude that a complete, lifecycle-based accounting
framework is necessary to accurately represent the full climate impact of AI infrastructure.
**Source location:** BLOOM: Sections 4.1 and 4.3 (pages 4–7); Greenpeace: Introduction and
Section "Energy Demand from AI Chipmaking" (pages 10–11, 16–22)
