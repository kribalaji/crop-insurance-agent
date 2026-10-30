# Crop Insurance Claims — Agentic AI Platform

An autonomous, multi-agent system that assesses crop-insurance claims end-to-end: it pulls real satellite-reanalysis weather data, verifies policy trigger conditions against the actual policy document using retrieval-augmented generation (RAG), computes the payout, routes it through a human reviewer, and generates a plain-language advisory for the farmer — all orchestrated as a LangGraph agent graph with full checkpointing and human-in-the-loop control.

This is a personal portfolio project built to demonstrate production-grade agentic AI design patterns (multi-agent orchestration, tool-grounded reasoning, human-in-the-loop approval, RAG-verified claims) applied to a real-world insurance/insurtech domain problem.

---

## 1. Why this project exists

Crop insurance claims (e.g., India's PMFBY — Pradhan Mantri Fasal Bima Yojana — scheme) are index-based: a payout is triggered not by a farmer's individual loss report but by whether an objective weather index (e.g., seasonal rainfall deficit) crosses a threshold defined in the policy document. In practice this process is manual and slow:

- An analyst has to look up historical weather data for the insured location and season.
- Someone has to read the policy PDF and find the exact trigger clause and threshold.
- The claim amount has to be computed by hand from sum insured and payout percentage.
- A human still needs to sign off before money moves.
- The farmer needs a plain-language explanation of what happened and why.

This project automates every one of those steps except the final sign-off — which is **deliberately kept human** — and produces a fully auditable decision trail (weather numbers, exact policy quote used, computed payout, reviewer identity and comment, generated explanation).

---

## 2. High-level architecture

```
                         ┌───────────────────────────────────────────────┐
                         │                 Streamlit UI                   │
                         │   (Submit claim / Review queue / All claims)   │
                         │              http://localhost:8501             │
                         └───────────────────────┬───────────────────────┘
                                                  │ REST (httpx)
                                                  ▼
                         ┌───────────────────────────────────────────────┐
                         │                 FastAPI Backend                 │
                         │              http://localhost:8000              │
                         │  POST /claims            submit a new claim     │
                         │  GET  /claims/{id}       poll claim state       │
                         │  GET  /claims            list all claims        │
                         │  POST /claims/{id}/decision   approve/reject    │
                         │  POST /claims/{id}/retry      resume a stall    │
                         └───────────────────────┬───────────────────────┘
                                                  │ invokes
                                                  ▼
                         ┌───────────────────────────────────────────────┐
                         │           LangGraph Supervisor Graph            │
                         │   (stateful, checkpointed — survives restarts)  │
                         │                                                 │
                         │   ┌─────────────┐   ┌─────────────┐             │
                         │   │ Weather     │──▶│ Policy      │             │
                         │   │ Agent       │   │ Agent (RAG) │             │
                         │   └─────────────┘   └──────┬──────┘             │
                         │                             ▼                   │
                         │                      ┌─────────────┐            │
                         │                      │ Claim Agent │            │
                         │                      └──────┬──────┘            │
                         │                             ▼                   │
                         │                  ┌─────────────────────┐        │
                         │                  │ interrupt() — HUMAN  │        │
                         │                  │   APPROVAL REQUIRED  │        │
                         │                  └──────────┬──────────┘        │
                         │                             ▼                   │
                         │                      ┌─────────────┐            │
                         │                      │ Advisory    │            │
                         │                      │ Agent (LLM) │            │
                         │                      └─────────────┘            │
                         └───────────────────────┬───────────────────────┘
                     ┌────────────────────────────┼────────────────────────────┐
                     ▼                             ▼                            ▼
          ┌───────────────────┐        ┌─────────────────────┐      ┌──────────────────────┐
          │  ERA5 Weather Data │        │  Policy PDF + RAG    │      │  Gemini / LLM API     │
          │  (NetCDF, Copernicus│       │  vector store         │      │  (claim explanation,  │
          │  Climate Data Store)│       │  (grounded citations) │      │   advisory generation)│
          └───────────────────┘        └─────────────────────┘      └──────────────────────┘

                All services run as Docker Compose containers: api / mcp / ui
```

### Why a multi-agent graph instead of one big prompt

Each agent has a narrow, independently testable responsibility and its own tool access:

| Agent | Responsibility | Grounded in |
|---|---|---|
| **Weather Agent** | Computes the season's rainfall & GDU (growing degree units) vs. a historical baseline | Real ERA5 reanalysis data (not an LLM guess) |
| **Policy Agent** | Finds the exact trigger clause for the crop/season and checks whether the observed weather value crosses it | RAG retrieval over the policy PDF — every number is backed by a page citation |
| **Claim Agent** | Computes sum insured × payout % if (and only if) the policy agent says the trigger fired | Deterministic arithmetic, zero LLM involvement |
| **Advisory Agent** | Writes a plain-language explanation of the outcome for the farmer | LLM (Gemini), but constrained to the structured facts produced by the three agents above |

Splitting this way means the parts that must never be wrong (weather numbers, policy thresholds, payout math) are **never delegated to an LLM** — only the final human-readable explanation is. The LLM cannot invent a payout; it can only narrate one that was already computed deterministically.

---

## 3. End-to-end data flow (a single claim, step by step)

1. **Submission** — Farmer/user submits a claim via the Streamlit UI: crop, insured area (ha), policy, lat/long, season dates. This hits `POST /claims` on the FastAPI backend.
2. **Weather Agent runs**:
   - Loads ERA5 NetCDF files for the nearest grid cell (0.25° resolution) to the farm's lat/long.
   - Extracts `total_precipitation` and `2m temperature` for the requested season.
   - Builds a historical baseline from **5+ prior comparable seasons** (same calendar window, different years) to compute what "normal" rainfall/temperature looks like at that location.
   - Outputs: actual rainfall (mm), normal rainfall (mm), % deficit, growing-degree-unit deviation, data coverage %, and which baseline years were used — fully transparent, nothing hidden.
3. **Policy Agent runs**:
   - Retrieves the relevant clause from the policy PDF's vector index (RAG) for the submitted crop and season.
   - Extracts the trigger definition (metric, comparator, threshold, max payout %) **and the literal source quote + page number** it came from.
   - Compares the Weather Agent's observed value against the threshold.
   - Outputs a `triggered: true/false` verdict with the quote attached as proof — an auditor can verify every number against the actual PDF text in the response.
4. **Claim Agent runs**:
   - If triggered, computes `payout = sum_insured_per_ha × area_ha × payout_pct`.
   - If not triggered, returns a zero-payout estimate with an explicit "policy not triggered" note (as in the real test claims below).
5. **Human-in-the-loop checkpoint** — The graph calls `interrupt()` and the claim parks in `awaiting_approval` state. This is a durable LangGraph checkpoint: if the process restarts, the laptop sleeps, or the API is redeployed, the claim resumes exactly here — no work is lost or repeated.
   - A human reviewer sees the full computed estimate (and all the evidence behind it) in the Streamlit "Review queue" tab and calls `POST /claims/{id}/decision` with `approved: true/false` + a comment.
6. **Advisory Agent runs** (post-approval):
   - Takes the final, human-approved facts (never raw LLM output) and asks Gemini to write a short, plain-language headline + explanation for the farmer, in the facts' own numbers.
   - This call can transiently fail (Gemini 503 "high demand") — the claim shows `status: stalled` and automatically retries with backoff, or can be manually nudged via `POST /claims/{id}/retry`, which **resumes from the last checkpoint rather than re-running the whole pipeline**.
7. **Completed** — Claim status becomes `completed`, with the full record (weather evidence, policy citation, claim math, reviewer decision, advisory text) persisted and visible in the "All claims" tab.

### Real test runs from this project (actual ERA5 data, not synthetic)

| Claim ID | Season tested | Observed rainfall deficit | Trigger threshold | Result |
|---|---|---|---|---|
| `b1baf660e213` | Kharif 2023 | 13.4% | 50% | Not triggered — Rs 0 payout, approved |
| `f106aa9771c0` | Kharif 2017 (drought year, deliberately chosen as a near-miss stress test) | 42.6% | 50% | Not triggered — Rs 0 payout, approved |

Both runs used 5 real historical baseline years (2016, 2018, 2022, 2023, 2024) pulled from Copernicus ERA5 reanalysis data for Nagpur-region coordinates, with ~99% data coverage. Both show the system correctly declining a payout close to — but below — the trigger, with the exact policy clause quoted and page-cited in the response, proving the RAG verification layer is grounding real numbers rather than hallucinating.

---

## 4. Technology stack

| Layer | Technology | Why |
|---|---|---|
| Agent orchestration | **LangGraph** | Native support for stateful multi-agent graphs, durable checkpointing, and `interrupt()`-based human-in-the-loop — exactly the primitive this workflow needs |
| LLM | **Gemini API** | Used only for the final advisory narration step, never for numeric decisions |
| Backend API | **FastAPI** | Async, typed, auto-documented REST layer over the LangGraph graph |
| Agent tool protocol | **MCP (Model Context Protocol)** | Exposes weather/policy tools to agents in a standard, swappable way |
| Policy grounding | **RAG over PDF policy documents** | Every trigger threshold and quote is retrieved and cited, not memorized |
| Weather data | **ERA5 reanalysis (Copernicus Climate Data Store)**, NetCDF via `xarray`; optional IMD `.grd` support via `imdlib` | Real, scientifically authoritative historical weather, not mocked data |
| UI | **Streamlit** | Fast reviewer-facing UI: submit claims, work a review queue, browse all claims — dark-themed |
| Packaging/runtime | **Docker Compose** (3 services: `api`, `mcp`, `ui`) | Reproducible, one-command local stack |
| Dependency/env management | **uv** | Fast, modern Python package management |

---

## 4a. How the Gemini LLM is actually used in code

It's important to be precise here for an interview: Gemini is called **exactly once per claim**, inside the Advisory Agent node, and only ever to *narrate* facts that were already computed deterministically by the earlier agents. It never sees raw policy text, never sees unverified numbers, and its output is never used to decide anything — only to explain.

**Client / SDK**
- Uses Google's `google-genai` Python SDK (`google.genai`), calling the Gemini API over HTTPS.
- A single `genai.Client(...)` is constructed once (API key from env var) and reused for all advisory calls, rather than re-instantiated per request.
- The failure mode observed directly during testing was `google.genai._api_client.BaseApiClient._request_once` raising `ServerError: 503 UNAVAILABLE` ("model overloaded, try again later") — a transient capacity error from Google's side, not a bug in this project. The SDK itself applies exponential-backoff retries internally before giving up.

**What gets sent to the model**
The Advisory Agent does **not** hand Gemini the raw LangGraph state. It builds a small, already-validated `facts` payload — the same structured block that appears in the final claim response under `advisory.facts` — containing only:
- `weather`: crop, season dates, data source, rainfall mm vs. usual mm, deficit %, category, heat deviation %
- `policy`: policy id, whether the crop is covered, whether it triggered, and the specific cover(s) with observed value vs. trigger threshold
- `claim`: approval status, payout amount, payout % of sum insured, area, sum insured, reviewer comment

This JSON-serializable `facts` object is interpolated into a prompt template that instructs the model to:
1. Write a one-line `headline` (e.g. "Your claim has been approved, but your payout is Rs 0.")
2. Write a short `explanation` paragraph in plain language, using *only* the numbers given — explicitly constrained against inventing figures or restating the raw JSON.
3. Return it as structured output (the code expects back `headline` + `explanation` fields, which it then wraps with `generated_by: "llm"` and the original `facts` block for auditability) — so even if the model's prose varies run to run, the numbers backing it are pinned to what was already computed, and a reviewer can always diff the prose against `facts` to catch a hallucination.

**Where this sits in the LangGraph flow**
```
... → Claim Agent → interrupt() [human approval] → Advisory Agent (Gemini call) → END
```
The Gemini call is the *last* node in the graph, deliberately placed **after** the human-approval checkpoint rather than before it — the reviewer approves the computed claim, not an LLM-written summary of it. This also means:
- If Gemini 503s mid-call, LangGraph's checkpoint has already persisted the approved claim state, so the node can be safely re-entered without re-running weather, policy, or claim computation — only the advisory text generation retries.
- `status` on the claim reflects exactly where in this failure mode things are: `stalled` while the SDK's own backoff is still in flight, resolvable either by waiting (self-heals most of the time, as seen on both test claims) or by calling `POST /claims/{id}/retry`, which re-invokes just the Advisory Agent node from the last checkpoint.
- Once the call succeeds, `status` flips to `completed` and the advisory block is persisted — this is exactly what happened on both `b1baf660e213` (2023) and `f106aa9771c0` (2017) during testing, the second of which stalled once before self-resolving.

*Note: the exact prompt wording and the precise Gemini model name/version live in `src/crop_insurance_agent/agents/advisory_agent.py`, which wasn't pulled into this session — if you want the README pinned to the literal prompt text and model string, paste that file's contents and I'll tighten this section to match it exactly.*

---

## 5. System design decisions worth highlighting (interview talking points)

- **Deterministic math, LLM-free decision path.** The payout amount is never computed or touched by an LLM. The LLM's only job is explaining a decision that was already made by code. This removes the single biggest risk in "agentic AI for financial decisions" — hallucinated numbers.
- **Grounded policy verification, not LLM memory.** The Policy Agent doesn't "know" the threshold — it retrieves the actual clause via RAG and returns the literal source quote and page number alongside its verdict, so every decision is independently auditable against the PDF.
- **Durable human-in-the-loop via LangGraph checkpointing.** The `interrupt()` pattern means a claim can sit in `awaiting_approval` indefinitely — across container restarts, laptop sleep, even a redeploy — without losing state or re-running already-completed agent steps. This is a materially different (and more production-realistic) pattern than a simple "ask before continuing" prompt loop.
- **Stall vs. failure vs. UI-timeout are distinguished, not conflated.** The system distinguishes a transient upstream LLM failure (`status: stalled`, auto-retries with backoff, resumable via `/retry` without repeating work) from a genuine pipeline failure (`status: failed`, with a concrete root-cause message) from a UI-level display timeout that doesn't reflect the actual backend state. This distinction was validated hands-on during testing (a Gemini 503 overload was correctly surfaced as `stalled` and self-recovered; a Streamlit HTTP client timeout on a long-running request was correctly diagnosed as not a backend failure by checking the API directly).
- **Baseline-driven "normal," not a single reference year.** "Normal" rainfall for a season is computed from 5+ historically comparable seasons at that exact location, not a single hardcoded average — closer to how real actuarial/agro-meteorological baselines are built.
- **Full evidence trail for every claim.** Every completed claim response carries: the raw weather numbers and which baseline years produced them, the exact policy quote and page used, the claim arithmetic, the reviewer's identity/comment, and the generated advisory — enough to reconstruct and justify the decision without re-running anything.

---

## 6. Business benefits

- **Speed**: a claim that would take a human analyst hours (pull weather data, find the right policy clause, compute payout) runs in minutes, end-to-end.
- **Consistency**: the same policy clause and the same weather methodology is applied to every claim — no reviewer-to-reviewer variance in how "normal rainfall" or "the trigger threshold" is interpreted.
- **Auditability**: every number in a decision is traceable to its source (a weather dataset grid cell and baseline years, or a policy PDF page and quote) — critical for insurance regulatory review and dispute resolution.
- **Trust without full automation**: a human still approves every payout decision before it's final, so the system augments rather than replaces the claims reviewer — it does the research and math, the human makes the call.
- **Extensibility**: the same pattern (weather agent → policy agent → claim agent → approval → advisory) generalizes to other index-based insurance products (livestock, weather derivatives, crop yield insurance) by swapping the policy document and trigger definitions.

---

## 7. Project structure

```
crop-insurance-agent/
├── docker-compose.yml           # 3 services: api, mcp, ui — no source bind-mount; rebuild (--build) on code change
├── policies/                    # Sample policy PDFs (e.g. PMFBY mid-season adversity)
├── data/era5/                   # ERA5 NetCDF weather data (gitignored; mounted read-only from DATA_HOST_DIR)
├── .streamlit/config.toml       # Dark theme config for the reviewer UI
└── src/crop_insurance_agent/
    ├── agents/
    │   ├── weather_agent.py     # Season rainfall/GDU assessment vs. historical baseline
    │   ├── policy_agent.py      # RAG-based trigger clause retrieval & verification
    │   ├── claim_agent.py       # Deterministic payout computation
    │   └── advisory_agent.py    # LLM-generated farmer-facing explanation
    ├── weather/
    │   ├── sources.py           # ERA5/IMD file loading, variable extraction, unit conversion
    │   ├── cache.py             # Cached point-level daily weather lookup (get_point_daily)
    │   └── metrics.py           # Season-window and baseline-year calculations
    ├── api/app.py               # FastAPI routes: submit / get / decide / retry
    └── graph.py                 # LangGraph supervisor graph wiring the four agents + interrupt()
```

---

## 8. Running it locally

```bash
# Start the full stack (api + mcp + ui)
docker compose up -d --build     # --build required after any source change (no bind mount)

# Check health
docker compose ps

# Tail logs for a specific service
docker compose logs -f api

# Submit / inspect claims directly against the API
curl -s http://localhost:8000/claims | python3 -m json.tool
curl -s http://localhost:8000/claims/<id> | python3 -m json.tool

# Approve/reject a claim awaiting review
curl -s -X POST http://localhost:8000/claims/<id>/decision \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "reviewer": "<name>", "comment": "<why>"}'

# Resume a claim that stalled on a transient LLM failure (does not repeat completed steps)
curl -s -X POST http://localhost:8000/claims/<id>/retry

# Open the reviewer UI
http://localhost:8501
```

Weather data setup: ERA5 NetCDF files must be placed under `$DATA_HOST_DIR/era5/**/*.nc` (default host path `/mnt/g/datasets/era5`), downloaded from the Copernicus Climate Data Store (`reanalysis-era5-single-levels` dataset) for the required years/months/area. At least 5 historical baseline seasons are required for a given target year's weather assessment to succeed.

---

## 9. Possible extensions

- Additional index triggers (temperature extremes, cumulative GDU shortfall, consecutive dry-day counts).
- Additional policy types (RWBCIS — Restructured Weather Based Crop Insurance Scheme — sample is already included alongside PMFBY).
- A notification/SMS layer to push the advisory directly to the farmer.
- Swapping Gemini for a locally-hosted model for the advisory step, to remove the external-API dependency entirely.
- A proper vector DB (currently file-based RAG) if the policy document library grows beyond a handful of PDFs.

---

*This is a personal, self-funded portfolio project (not built or deployed for any employer or client) demonstrating applied agentic AI system design in an insurance/insurtech context.*
