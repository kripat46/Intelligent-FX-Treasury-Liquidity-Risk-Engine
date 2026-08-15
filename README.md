# Wise-Style Liquidity & Fraud Intelligence Engine

![CI](https://github.com/kripat46/Intelligent-FX-Treasury-Liquidity-Risk-Engine/actions/workflows/ci.yml/badge.svg)

A portfolio-grade simulation of the backend intelligence layer behind a
multi-currency remittance platform. It combines a **real-time
fraud/AML risk engine** (MLE track) with a **treasury liquidity optimization
engine** (DS/quant track), surfaced through a live **Streamlit control tower**.

## Why this architecture

Real fintech risk platforms are built from two decoupled but cooperating
systems:

1. A **behavioral risk engine** that scores each transaction independently,
   in real time, using engineered velocity/behavioral features -> this is a
   supervised learning + imbalanced classification problem.
2. A **treasury/liquidity engine** that manages currency float pools as an
   *inventory problem under uncertainty* -> this is a stochastic optimization
   problem (newsvendor / (s, S) inventory theory), not a classification
   problem. Conflating the two is the most common design mistake in "fraud
   detection" portfolio projects; keeping them as separate services with a
   shared data contract (the transaction stream) is what makes this project
   look production-grade rather than tutorial-grade.

```
┌─────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────┐
│ 1. Synthetic Stream │ ---> │ 2. Fraud Risk Engine    │ ---> │ 4. Control Tower      │
│    Generator        │      │   (XGBoost + feature    │      │   (Streamlit)         │
│  transactions.csv   │      │    pipeline, 3-tier     │      │  - live tx feed       │
└──────────┬──────────┘      │    action policy)       │      │  - confidence plots   │
           │                 └─────────────────────────┘      │  - float gauges       │
           │                                                  │  - rebalancing alerts │
           ▼                                                  └───────────▲───────────┘
┌──────────────────────────┐                                              │
│ 3. Liquidity Optimizer   │ -------------------------------------------- ┘
│   (s,S) newsvendor model │
│   + dynamic fee pricing  │
└──────────────────────────┘
```

## Components
 
| # | File | Concern | Core technique |
|---|------|---------|-----------------|
| 1 | `src/data_generator.py` | Synthetic, skewed, fraud-labeled transaction stream (offline analysis scale) | Log-normal amount modeling, injected structuring/velocity/spike patterns |
| 2 | `src/feature_engineering.py` | Point-in-time-safe behavioral features (batch/offline) | Rolling windows via `groupby().rolling()`, robust z-scores, geo-mismatch flags |
| 3 | `src/fraud_model.py` | Risk scoring + 3-tier action policy, model training & persistence | XGBoost, `scale_pos_weight`, PR-curve threshold tuning for recall, latency-instrumented inference |
| 4 | `src/liquidity_optimizer.py` | Float pool sizing, dynamic pricing, **live pool evolution** | Newsvendor critical-ratio optimization (`scipy.optimize`), `LivePoolBook` (s, S) tick-by-tick simulation |
| 5 | `src/streaming_engine.py` | **Incremental, leak-free live feature computation + live transaction generator** | Bounded rolling state (deques) updated per-transaction, no batch dataframe |
| 6 | `src/bootstrap.py` | Self-contained cold-start: generate + train + solve in one call | Demo-scale config tuned for ~5-10s bootstrap |
| 7 | `dashboard/app.py` | Live visualization + orchestration | Streamlit + Plotly; owns the session's live state (feature store, simulator, pool book) |
| 8 | `api/main.py` | **Model serving, decoupled from the dashboard** | FastAPI + Pydantic validation, shares the same persisted engine and `LivePoolBook` |
| 9 | `src/train_with_mlflow.py` | **Experiment tracking** | MLflow: params, metrics, confusion matrix, and model artifact logged per run |
| 10 | `tests/` | Correctness guarantees for the above | pytest, 22 tests incl. full API integration via `TestClient` |
| 11 | `api/Dockerfile`, `dashboard/Dockerfile`, `mlflow/Dockerfile`, `docker-compose.yml` | Containerization | Lean, service-specific images; compose for local multi-container runs |
| 12 | `k8s/*.yaml` | Orchestration | Deployment/Service/HPA per service, shared ConfigMap; see `k8s/README.md` for scaling caveats |
| 13 | `.github/workflows/ci.yml` | CI/CD | lint → test → build, on every push/PR |
 
## Key modeling decisions
 
**Fraud as a 3-tier action policy, not a 3-class classifier.**
Real compliance systems don't train a model to predict "Blocked" directly —
label scarcity for "should have been blocked" is worse than for "is fraud",
and a 3-class target throws away the ordinal, monotonic structure of risk
(Blocked risk ⊃ Flagged risk). Instead we train **one binary fraud
probability model**, then apply two probability thresholds (`t_flag`,
`t_block`) tuned off the precision-recall curve. This is both more
statistically defensible and mirrors how real risk engines expose a single
calibrated score to a downstream rules/policy layer.
 
**Threshold selection minimizes false negatives, not accuracy.**
In fraud, a false negative (missed fraud) costs orders of magnitude more
than a false positive (friction on a legitimate user). We select `t_flag` by
walking the PR curve for the **recall level a compliance team would set as a
SLA** (e.g. recall ≥ 0.90) and taking the threshold that maximizes precision
subject to that recall floor — not the threshold that maximizes F1 or
accuracy, which would systematically under-flag rare fraud.
 
**Liquidity as a newsvendor / (s, S) inventory problem.**
A currency float pool is functionally an inventory of a perishable resource
(uninvested cash has a real cost of capital) facing stochastic daily demand
(net outflow). The classic trade-off between overage cost (holding cash) and
underage cost (stockout delaying a customer transfer, plus reputational
cost) is exactly the newsvendor problem, whose optimal solution is a closed
form: order up to the quantile of demand given by the **critical ratio**
`Cu / (Cu + Co)`. We use this closed form as the analytical anchor and
`scipy.optimize.minimize_scalar` to solve the total-expected-cost
minimization directly (so it's still legitimate to call this an
"optimization model" rather than a lookup formula), and confirm the two
agree.
 
**Dynamic pricing is scarcity-indexed, not fraud-indexed.**
The fee multiplier is a logistic function of *how close the pool is to its
reorder point*, independent of the fraud engine. This keeps the treasury
incentive mechanism auditable on its own terms (regulators and finance teams
need to be able to explain a fee change purely in terms of liquidity risk).
 
## The dashboard is genuinely live.
 
An earlier version of this dashboard read a pre-scored CSV and animated
through it. It's been replaced with a real (if simulated) streaming pipeline:
 
* **`src/streaming_engine.py`** — `StreamingFeatureStore` computes every
  feature (rolling velocity, personalized z-score, device fan-out) from
  bounded in-memory rolling state **as each transaction arrives**, one at a
  time, the way a real online feature store would — not by slicing a
  batch dataframe. `LiveTransactionSimulator` emits transactions
  one-by-one, injecting fraud bursts (structuring, device rings, volume
  spikes) probabilistically and live rather than pre-planting a static
  fraud layer.
* **`src/liquidity_optimizer.py`: `LivePoolBook`** — currency pools
  actually deplete from live transaction outflows each tick and actually
  replenish under an (s, S) order-up-to policy after a simulated lead
  time. Reorder-point breaches and rebalancing alerts in the dashboard are
  computed from this live state, not hardcoded.
* **`src/bootstrap.py`** — generates data, trains the model, and solves
  the liquidity pools from scratch in ~5-10 seconds, cached via
  `st.cache_resource`. This means the app is fully self-contained: deploy
  it to a fresh container with nothing pre-committed and it builds its own
  world on first load.
## Quickstart (local)
 
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
# first load takes ~5-10s to bootstrap (generate data, train model, solve pools)
# after that, use the sidebar toggle to start the live simulation
```
 
For the larger, full-scale offline analysis (the numbers quoted in this
README) rather than the fast demo-scale bootstrap the dashboard uses:
 
```bash
python src/data_generator.py            # full-scale synthetic set (~140k rows)
python src/fraud_model.py               # trains + persists data/fraud_engine.joblib
python src/liquidity_optimizer.py       # writes data/liquidity_state.json
```
 
## Deploying to Streamlit Community Cloud
 
1. Push this repository to GitHub (the `data/` directory is gitignored —
   that's intentional; the app builds it on first load).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   and set the entry point to `dashboard/app.py`.
3. Deploy. First load will take ~10-15s while it bootstraps; subsequent
   loads reuse the cached model/data for the life of the container.
No secrets, no external services, no database — the whole system runs
inside the Streamlit container.
 
---
 
## MLOps stack
 
The dashboard is one *consumer* of the model; a real platform needs the
model servable independently of whether anyone has a browser open, needs
its training runs tracked and comparable, and needs a repeatable path to
containerized/orchestrated deployment. This project has all four pieces:
 
| Concern | What's here |
|---|---|
| **Serving** | `api/main.py` — FastAPI app exposing `/score`, `/score/batch`, `/liquidity/state`, `/liquidity/tick`, `/health`, `/model/metrics`. Same `FraudDetectionEngine` and `LivePoolBook` the dashboard uses — one shared artifact layer, two independent consumers. |
| **Experiment tracking** | `src/train_with_mlflow.py` — logs hyperparameters, offline holdout metrics, the confusion matrix, and the trained model artifact to MLflow. Kept as a separate entrypoint from `fraud_model.py` so the core engine has no MLflow dependency. |
| **Containerization** | `api/Dockerfile`, `dashboard/Dockerfile`, `mlflow/Dockerfile` — three lean, service-specific images (the API image doesn't ship Streamlit; the dashboard image doesn't ship FastAPI). |
| **Orchestration** | `docker-compose.yml` for local multi-container runs; `k8s/` manifests (Deployment/Service/HPA per service + shared ConfigMap) for cluster deployment — see `k8s/README.md` for a documented scaling caveat before you raise replica counts. |
| **CI/CD** | `.github/workflows/ci.yml` — lint (ruff) → test (pytest, 22 tests) → build both Docker images, on every push/PR. |
| **Testing** | `tests/` — 22 pytest tests covering feature-leakage safety, threshold-policy monotonicity, newsvendor closed-form vs. numerical agreement, incremental streaming-feature correctness, live pool depletion/replenishment, and full API integration tests via `TestClient`. |
 
### Running the API
 
```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
# interactive docs at http://localhost:8000/docs
```
 
```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "Transaction_ID": "TXN000001", "User_ID": 42, "Source_Currency": "USD",
  "Target_Currency": "INR", "Amount_USD": 9700.0, "Device_IP": "24.1.1.1"
}'
```
 
### Running everything with Docker Compose
 
```bash
docker compose up --build
# API:        http://localhost:8000/docs
# Dashboard:  http://localhost:8501
# MLflow UI:  http://localhost:5000
```
 
### Running the test suite / lint locally (what CI runs)
 
```bash
pip install -r requirements-dev.txt -r requirements-api.txt
ruff check src/ api/ dashboard/ tests/
pytest tests/ -v
```
 
### Tracking a training run with MLflow
 
```bash
pip install -r requirements-dev.txt
python src/train_with_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs at http://localhost:5000
```
 
### Deploying to Kubernetes
 
See `k8s/README.md` — includes the image-push steps and, importantly, an
explanation of why the API's HorizontalPodAutoscaler is deliberately
pinned to 1 replica until the streaming feature store is backed by shared
state.
## What's verified vs. not
 
Being direct about this rather than letting a reader (or an interviewer)
find out the hard way:
 
**Actually run and verified (across development and local deployment):**
- The full pytest suite (22/22 passing) — feature engineering, threshold
  policy, liquidity optimizer, streaming engine, and the FastAPI service
  via `TestClient` (real HTTP-shaped requests, not mocked).
- The FastAPI app, end-to-end, including a near-threshold transaction
  correctly scored and flagged, and a validation error correctly
  returning 422.
- MLflow tracking — a real run was logged and independently queried back
  via `MlflowClient` (params, metrics, and the model artifact all
  confirmed present, not just "no exception was thrown").
- The Streamlit dashboard, via Streamlit's official `AppTest` harness —
  cold-start bootstrap, manual tick advances, tab switches, and session
  reset all execute with zero exceptions.
- `ruff check` passes clean with a deliberately scoped rule set (see
  `pyproject.toml`).
- **`docker build` + `docker compose up`, run locally on a real machine
  (Docker Desktop / Windows)**: all three services (`api`, `dashboard`,
  `mlflow`) build and reach a healthy state, with the API's `/health`
  endpoint and MLflow's UI both confirmed reachable. One real issue was
  found and fixed this way that the sandbox's YAML-only validation could
  never have caught: MLflow's default multi-worker startup raced to
  initialize the same SQLite schema concurrently, crash-looping a few
  times before stabilizing — fixed by pinning `--workers 1`. A second
  issue — an unnecessary `depends_on: mlflow` on the `api` service — was
  also removed after it caused a flaky MLflow startup to cascade into
  blocking the entire stack, even though `api` doesn't call MLflow at
  runtime. Both are exactly the kind of integration bug that only shows
  up when you actually run the thing, not when you review the YAML.
**Written and validated for correctness, but not yet executed against
live infrastructure:**
- `.github/workflows/ci.yml` — the lint/test steps it runs were verified
  locally (that's exactly what runs in CI), and the workflow YAML is
  valid, but confirm it's gone green on the repo's **Actions** tab before
  treating it as proven.
**Actually deployed to a live Kubernetes cluster** (Docker Desktop's
built-in `kind`-based cluster, single node): `kubectl apply -f k8s/`
against a real cluster, `treasury-api` and `treasury-dashboard` both
reached `1/1 Running`, and the dashboard was reached and confirmed
working through `kubectl port-forward`. This surfaced three real issues
that YAML validation alone could not have caught, each worth naming
rather than glossing over:
 
1. **Stale placeholder image references.** The Deployment manifests
   initially pointed at `your-registry/treasury-api:latest` — a
   placeholder left over from before local image names were substituted
   in. `kubectl describe pod` surfaced the actual cause immediately
   (`pull access denied... repository does not exist`), and the fix was a
   one-line correction to the `image:` field in each Deployment. The
   general lesson: `kubectl describe pod` → **Events** section is the
   fastest path to ground truth on any pod stuck in `ImagePullBackOff` —
   guessing at causes without it wastes time.
2. **Docker Desktop's `kind`-based Kubernetes doesn't share an image
   store with `docker build`/`docker compose build` by default,** and
   also doesn't share state with a separately-installed standalone `kind`
   CLI (`kind get clusters` returns nothing even though Docker Desktop's
   own dashboard shows an active node) — the two are different
   implementations that happen to share a name. The working fix: `docker
   save` each image to a tarball, `docker cp` it into the cluster's node
   container (`desktop-control-plane`, itself just a Docker container),
   then `docker exec ... ctr -n k8s.io images import` to load it directly
   into the node's containerd image store, bypassing both `docker`'s and
   `kind`'s normal image-sharing paths entirely.
3. **A CLI syntax gotcha**: `ctr -n=k8s.io` (with `=`) fails silently
   with a cryptic `No help topic for '.io'` rather than a clear "invalid
   flag" error; `ctr -n k8s.io` (space-separated) is the correct form for
   this version of `ctr`.
None of these three issues are bugs in this project's own code — they're
all platform/tooling friction specific to Docker Desktop's Kubernetes
integration on Windows. But finding and fixing them is exactly the kind
of debugging trail worth being able to describe in an interview, and a
considerably stronger answer than either "I didn't try" or a suspiciously
clean "it just worked."

