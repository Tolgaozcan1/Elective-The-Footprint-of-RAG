# Corpus Sources

Five documents form the evaluation corpus. They represent a deliberate mix of
categories — academic peer-reviewed research, international organization policy analysis,
tech-company sustainability reporting, independent research-institute analysis, and NGO
investigative reporting — reflecting the breadth of perspectives on AI energy consumption
and carbon footprint.

---

## 1. BLOOM — Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model

| Field | Value |
|-------|-------|
| **Title** | Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model |
| **Authors** | Alexandra Sasha Luccioni, Sylvain Viguier, Anne-Laure Ligozat |
| **Year** | 2022 (arXiv) / 2023 (JMLR Vol. 24) |
| **Source URL** | https://jmlr.org/papers/volume24/23-0069/23-0069.pdf |
| **Category** | Academic paper (peer-reviewed) |

**Summary:** The paper applies a Life Cycle Assessment (LCA) framework to quantify the
carbon footprint of BLOOM, a 176-billion parameter open-source multilingual language model,
estimating 24.7 tonnes of CO₂eq from dynamic power consumption alone and 50.5 tonnes
when equipment manufacturing and idle consumption are included. It also conducts an
empirical study of inference energy consumption via an API deployment, arguing that
idle and embodied emissions substantially undercount the true footprint when omitted.

---

## 2. IEA — World Energy Outlook Special Report: Energy and AI

| Field | Value |
|-------|-------|
| **Title** | World Energy Outlook Special Report — Energy and AI |
| **Author** | International Energy Agency (IEA) |
| **Year** | April 2025 |
| **Source URL** | https://iea.blob.core.windows.net/assets/de9dea13-b07d-42c5-a398-d1b3ae17d866/EnergyandAI.pdf |
| **Category** | International organization report |

**Summary:** This comprehensive 304-page IEA special report analyses global electricity
demand from data centres (415 TWh in 2024, projected to roughly double to 945 TWh by
2030), explores how AI adoption intersects with energy security, grid infrastructure, and
emissions, and assesses how AI can simultaneously be deployed to optimise the energy
sector. It is the most expansive cross-sectoral quantitative analysis of the energy–AI nexus
published to date.

---

## 3. Google — 2025 Environmental Report

| Field | Value |
|-------|-------|
| **Title** | Google 2025 Environmental Report |
| **Author** | Google LLC |
| **Year** | June 2025 (FY2024 data) |
| **Source URL** | https://www.gstatic.com/gumdrop/sustainability/google-2025-environmental-report.pdf |
| **Category** | Tech company sustainability report |

**Summary:** Google's tenth annual environmental report covers FY2024 performance,
reporting a 12% reduction in data centre energy emissions despite a 27% increase in
electricity consumption, driven by clean energy procurement (8 GW contracted, 2.5 GW
brought online) and hardware efficiency gains (the Ironwood TPU is nearly 30× more
power-efficient than Google's first Cloud TPU from 2018). The report also details
AI-enabled environmental applications — including flood forecasting, fuel-efficient routing,
and wildfire detection — and acknowledges that supply-chain (Scope 3) emissions rose
even as operational emissions fell.

---

## 4. EPRI — Powering Intelligence: Analyzing Artificial Intelligence and Data Center Energy Consumption

| Field | Value |
|-------|-------|
| **Title** | Powering Intelligence: Analyzing Artificial Intelligence and Data Center Energy Consumption |
| **Author** | Electric Power Research Institute (EPRI) |
| **Year** | May 2024 |
| **Source URL** | https://www.wpr.org/wp-content/uploads/2024/06/3002028905_Powering-Intelligence_-Analyzing-Artificial-Intelligence-and-Data-Center-Energy-Consumption.pdf |
| **Category** | Independent research institute report |

**Summary:** EPRI's white paper develops four US data centre load-growth scenarios for
2023–2030 (ranging from 3.7% to 15% annual growth), projecting data centres will consume
4.6%–9.1% of US electricity by 2030, up from roughly 4% in 2023. The report highlights the
geographic concentration of demand (15 states account for 80% of national load, led by
Virginia), documents that a ChatGPT query requires approximately 10× the electricity of a
traditional Google search (2.9 Wh vs. 0.3 Wh), and identifies energy efficiency, grid
collaboration, and clean energy sourcing as the three essential strategies for managing
rapid data centre expansion.

---

## 5. Greenpeace — Chipping Point: Tracking Electricity Consumption and Emissions from AI Chip Manufacturing

| Field | Value |
|-------|-------|
| **Title** | Chipping Point: Tracking Electricity Consumption and Emissions from AI Chip Manufacturing |
| **Author** | Greenpeace East Asia |
| **Year** | April 2025 |
| **Source URL** | https://www.greenpeace.org/static/planet4-eastasia-stateless/2025/04/5011514f-greenpeace_chipping_point.pdf |
| **Category** | NGO report |

**Summary:** Using a bottom-up methodology, the report estimates that global electricity
consumption from AI chip manufacturing (covering Nvidia A100, H100, H200, B100/200 and
AMD MI300X) increased by more than 350% from 218 GWh in 2023 to approximately 984 GWh
in 2024, with associated CO₂eq emissions rising more than 4.5-fold to over 453,600 metric
tons. The report focuses on the overlooked climate impact of upstream semiconductor
manufacturing concentrated in fossil-fuel-heavy East Asian grids, and calls for tech
companies and chip manufacturers to commit to 100% renewable energy across supply
chains by 2030.

---

## AI Use Declaration

In line with the course's emphasis on transparent and responsible AI use, this
section documents how AI assistants contributed to preparing the contents of
this `dataset/` directory.

**Tools used:** Anthropic Claude (web chat) for source identification and
methodological discussion; Anthropic Claude Code (CLI) for reading the PDFs,
computing corpus statistics, and drafting the evaluation questions and
reference answers.

**AI-assisted steps:**
- Candidate sources were identified through AI-assisted web search and
  shortlisted manually based on diversity criteria (academic / international
  organization / tech company / independent research / NGO).
- PDFs were downloaded directly from the publishers' official sources.
- Corpus statistics in `corpus_stats.md` were computed via `pypdf` text
  extraction.
- One- to two-sentence summaries above were drafted by Claude Code after
  reading each PDF.
- Evaluation questions in `eval_questions.md` were drafted by Claude Code by
  reading each PDF; the question count distribution and type mix were
  specified manually.

**Human verification:** Key numerical claims in the evaluation questions
(notably: the USD 12 trillion AI-related share of S&P 500 market-cap growth in Question 5, the 10/30/60
AI energy footprint split in Question 11, and the 218→984 GWh chipmaking
growth and 37,238 GWh 2030 projection in Question 12) were independently
verified against the source PDFs and confirmed to appear verbatim.
