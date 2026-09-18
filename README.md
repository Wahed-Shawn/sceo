# GridWise — Smart Campus Energy Optimization (LLM-Assisted)

BUP CSE Fest 2026 · Hackathon · Online Preliminary

An HTTP service that interprets natural-language operator notes with an LLM,
deterministically validates them, and solves an LP optimization to return a
cost-minimal 24-hour battery dispatch schedule.

## Pipeline

```
operator notes ──► LLM interpretation ──► deterministic guardrails ──► PuLP LP optimizer ──► JSON response
  (1-3 notes)        (OpenAI chat)           (schema/type/value fixups)   (minimize grid cost)
```

- **LLM role**: The generative model produces the `directive_interpretation`
  array (one entry per note: `note_index`, `applies`, `directive_type`,
  `structured_adjustment`, `explanation`). This is mandatory per the challenge
  rules.
- **Guardrails**: Raw model output is normalized/validated before optimization
  so invalid model output can never inject a bogus constraint.
- **Optimizer**: PuLP LP (CBC) minimizing `Σ grid_kwh[h] * tariff[h]` subject to
  energy balance, effective-solar caps, battery bounds/rates/transitions,
  reserve floors, grid caps and no-charge/no-discharge windows, and end-of-day
  battery neutrality.

## Quickstart (fresh environment)

Prerequisites: Python 3.11+. Uses only the submitted repository.

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

# 2. Configure the LLM (required for the challenge interpretation path)
#    Google Gemini (validated with this service — fast "lite" models):
$env:OPENAI_API_KEY = "AIza... or AQ..."
$env:OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:GRIDWISE_MODELS = "gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-2.5-flash-lite"
#    or OpenAI:
# $env:OPENAI_API_KEY = "sk-..."
# $env:GRIDWISE_MODEL = "gpt-4o-mini"

# Optional: any OpenAI-compatible endpoint (Ollama, vLLM, Azure-compatible)
# export OPENAI_BASE_URL="http://localhost:11434/v1"

# 3. Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Resilience fallback**: if `OPENAI_API_KEY` is unset or a model call fails,
> the service automatically falls back to a deterministic rule-based parser so
> it stays up during judges. The LLM remains the primary interpretation path.
> Set `GRIDWISE_LLM_MODE=rules` to force the fallback for offline testing.

## Environment variables

| Variable           | Purpose                                          | Default      |
|--------------------|--------------------------------------------------|--------------|
| `OPENAI_API_KEY`   | OpenAI (or compatible) chat API key              | (unset)      |
| `OPENAI_BASE_URL`  | OpenAI-compatible base URL (local model allowed) | (unset)      |
| `GRIDWISE_MODEL`   | Single model id for note interpretation          | `gpt-4o-mini`|
| `GRIDWISE_MODELS`  | Comma-separated cascade; try in order, switch on timeout/failure | (none) |
| `GRIDWISE_LLM_TIMEOUT` | Seconds to wait per model before switching  | `3.0`        |
| `GRIDWISE_LLM_MODE`| `openai` (primary) or `rules` (fallback-only)    | `openai`     |
| `PORT`             | uvicorn port (used by deploy scripts)            | 8000         |

`GRIDWISE_MODELS` overrides `GRIDWISE_MODEL`. Example validated on Gemini
(free tier): `gemini-flash-lite-latest → gemini-3.5-flash-lite →
gemini-2.5-flash-lite`, each returning a correct interpretation in ~2 s. Larger
"thinking" models (e.g. `gemini-3.6-flash`) are not recommended: they can spend
30-45 s thinking and blow the per-request budget.

No secrets are committed. Never commit keys; supply them via environment.

## Endpoints

### `GET /health`

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}   (ready within 60s of start)
```

### `POST /optimize-energy`

Body schema (24 `hours` entries, hour 0..23; `operator_notes` 1-3 strings):

```jsonc
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast."
  ],
  "hours": [ { "hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6 }, "...24 entries total..." ],
  "battery": {
    "capacity_kwh": 220, "initial_energy_kwh": 110, "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50
  }
}
```

Full curl example (one of the public samples):

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  --data @BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
```

## Running the public-sample regression

Validates the whole pipeline (interpretation, directive application, plan
validity, totals, and cost-vs-reference-optimal) for all 10 samples. Uses the
deterministic fallback for a key-free run:

```bash
GRIDWISE_LLM_MODE=rules python run_samples.py
```

Expected: `10 cases, 10 passed.` with each case cost matching the reference
optimal within 0.01 BDT. Equivalent optimal schedules (different hourly action
patterns) are intentionally accepted; only the optimal-cost value is compared.

## Response contract

```jsonc
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    { "note_index": 0, "applies": true, "directive_type": "solar_reduction",
      "structured_adjustment": { "hours": [12, 13], "factor": 0.25 },
      "explanation": "..." },
    { "note_index": 1, "applies": false, "directive_type": "no_op",
      "structured_adjustment": null, "explanation": "..." }
  ],
  "hourly_plan": [
    { "hour": 0, "grid_kwh": 90, "solar_used_kwh": 0,
      "battery_action": "idle", "battery_kwh": 0, "battery_energy_after_kwh": 110 }
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365,
  "peak_grid_kwh": 175,
  "plan_summary": "Applied operator directives (...) and optimized battery dispatch ..."
}
```

Directive semantics (start-inclusive / end-exclusive time windows; hours are
the 24 military whole hours; `factor` is the usable solar fraction remaining,
so an "80% reduction" is `factor = 0.2`):

| directive_type         | structured_adjustment                                    |
|------------------------|----------------------------------------------------------|
| `solar_reduction`      | `{ "hours": [...], "factor": float 0..1 }`               |
| `minimum_battery_reserve` | `{ "hours": [...], "minimum_energy_kwh": float }`     |
| `no_charge_window`     | `{ "hours": [...] }`                                     |
| `no_discharge_window`  | `{ "hours": [...] }`                                     |
| `max_grid_window`      | `{ "hours": [...], "max_grid_kwh": float }`              |
| `no_op`                | `applies: false`, `structured_adjustment: null`          |

## Docker fallback (judge reproducibility)

The image is self-contained (API + web UI + public samples + regression runner),
with no baked-in credentials.

```bash
# 1. Build (no credentials baked in)
docker build -t gridwise .

# 2. Run the service
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... gridwise
curl -s http://localhost:8000/health        # -> {"status":"ok"}

# 3. Run the public-sample regression inside the same image (key-free)
docker run --rm -e GRIDWISE_LLM_MODE=rules gridwise python run_samples.py
#    -> 10 cases, 10 passed.  (expected)
```

The key must be supplied at run time via `-e OPENAI_API_KEY=...`; it is never in
the image.

## Deploy so judges can call it (step by step)

You need a public HTTPS URL. The service needs the LLM key at the platform as a
secured env var; the key is never in the repo.

### 1. Push the code (No Docker needed on your machine)

```bash
cd C:\Users\Wahed Shawn\Desktop\hackathon\testing
git init
git add .
git commit -m "GridWise — BUP CSE Fest 2026 submission"
```
Create a repo on GitHub (`github.com/new`, e.g. `gridwise-hackathon`), then:
```bash
git remote add origin https://github.com/YOU/gridwise-hackathon.git
git branch -M main
git push -u origin main
```
`.gitignore` already excludes `.env`, so your live API key does not leave your
machine.

### 2a. Deploy on Render (recommended — free, NO Docker required)

**Zero-config path (uses the included `render.yaml` blueprint):**

1. Dashboard.render.com → **New → Blueprint** → connect your GitHub repo.
2. Render reads `render.yaml`, creates the web service, and prompts for
   `OPENAI_API_KEY` as a secret (enter your Gemini key once — it is never
   stored in the repo).
3. Deploy → wait for "Live" → you get `https://gridwise.onrender.com`.

**Or manual without Docker:** New → **Web Service** → connect repo →
Environment **Python** → Build `pip install -r requirements.txt` → Start
`uvicorn main:app --host 0.0.0.0 --port $PORT` → add env vars →
Deploy. No local Docker install needed in any case.

Required env vars (all of these):
   - `OPENAI_API_KEY` (your Gemini key; without it every request falls back
     silently to the rules parser)
   - `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`
   - `GRIDWISE_MODELS=gemini-flash-lite-latest,gemini-3.5-flash-lite,gemini-2.5-flash-lite`
   - `GRIDWISE_LLM_TIMEOUT=5`

Verify once live: `curl https://gridwise.onrender.com/health` → `{"status":"ok"}`,
open `https://gridwise.onrender.com/` (console UI), and submit that URL.

Free-tier note: Render sleeps the service after ~15 min idle. Ping it once right
before judging (or open `/docs`) and it wakes in ~30-60 s (within the health
readiness window). Keep it warm during the 4 h window by sending a health GET
every 10 min; or temporarily upgrade to the $7 instance.

### 2b. Alternatives (also no Docker needed)

- **Railway** (`railway.app`): Import from GitHub → it auto-detects Python →
  add the same env vars. Start command
  `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Fly.io**: `fly launch`, then `fly secrets set OPENAI_API_KEY=...`
- **Google Cloud Run**: `gcloud run deploy` — logical pick since the LLM is
  Google; set env vars under *Variables & Secrets*.

The repo's `Dockerfile` exists for the Docker-based scoring/debugging path
(built remotely by Render/Railway/Fly — you never need Docker installed locally
unless you want to test the image yourself).

### 3. Post-deploy checklist (rubric)

- `curl <URL>/health` returns `{"status":"ok"}` within 60 s of cold start.
- `curl <URL>/status` shows `"llm_configured": true` and the model cascade.
- `POST <URL>/optimize-energy` with a public sample returns a valid JSON plan
  and `directive_interpretation[].explanation` reads like LLM text (not
  "Deterministic fallback...").
- Repo contains no secrets (`git grep AQ.Ab8 -- .` should match nothing).

## Performance & reliability

- LP solves in <50 ms; the LLM call dominates. p95 well under the 5 s target.
- Per-request contract timeout is 30 s (FastAPI/uvicorn default).
- Model-provider failure is caught and routed to the deterministic fallback; the
  service never 500s on malformed model output, and no stack traces or secrets
  are returned to clients.

## File layout

| File            | Purpose                                          |
|-----------------|--------------------------------------------------|
| `main.py`       | FastAPI app, `/health`, `/optimize-energy`       |
| `models.py`     | Pydantic request/response schemas                |
| `interpreter.py`| LLM prompt + OpenAI-compatible call + fallback   |
| `guardrails.py` | Deterministic validation of LLM output           |
| `optimizer.py`  | PuLP LP formulation + rounding/normalization     |
| `run_samples.py`| Public-sample regression (judge-style checks)    |
| `static/index.html` | Web console UI (served at `/`)                 |
| `Dockerfile`    | Containerized fallback deployment                |
| `render.yaml`   | Render blueprint (no-Docker deploy)              |
| `.python-version` | Pins Python 3.12 for the platform runtime     |

## Known limitations

- No battery efficiency/round-trip loss is modeled (the problem does not define
  one); the schedule follows the stated energy-balance equation.
- Grid export/sell-back is not modeled; `grid_kwh >= 0`.
- Multiple overlapping directives of the same type combine conservatively
  (reserves/`factor`s take the tightest bound).
- The rule-based fallback is a resilience layer only; it does not cover every
  possible paraphrase. The LLM is the intended interpreter.