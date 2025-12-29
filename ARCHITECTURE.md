# Architecture Guide (Intuitive)

This repository is an end-to-end anomaly detection system for microservices.

At a high level:

1. **Collect**: grab log/event timestamps from Splunk, and metrics from Prometheus.
2. **Combine**: merge those sources into one timeline per service.
3. **Prepare**: turn “rows in a table” into a dense tensor the model can learn from.
4. **Learn**: train a model to reconstruct “normal behavior”.
5. **Detect**: on new data, reconstruction error becomes an anomaly score.
6. **Evaluate**: inject synthetic anomalies to compare preprocessing approaches.

The rest of this document walks you through the repo like you’d explain it to a teammate.

---

## Repo layout (what lives where)

Think of the repo as three layers:

- **Data gathering layer** (pulls raw info)
  - `src/ml_monitoring_service/data_gathering/`
    - `get_splunk_data.py`: fetches/parses Splunk results into JSON
    - `get_prometheus_data.py`: fetches Prometheus metrics
    - `combine.py`: merges Splunk + Prometheus into a combined dataset

- **ML/service layer** (turns data into model signals)
  - `src/ml_monitoring_service/`
    - `data_handling.py`: validation, preprocessing, tensor conversion, windowing
    - `model.py`: the actual neural network architecture
    - `anomaly_detector.py`: training loop, thresholding, detection logic
    - `main.py`: orchestration (scheduled training + inference) and one-shot CLI mode

- **Evaluation + tests** (safety rails and comparisons)
  - `tests/`: unit tests
  - `tests/resources/`: small fixtures used by tests
  - `src/ml_monitoring_service/evaluation/`: synthetic anomaly injection utilities
  - `scripts/synthetic_benchmark.py`: compares preprocessing approaches using synthetic anomalies

Output and experiment tracking:

- `output/`: produced datasets, models, plots/graphs, result summaries
- `mlruns/`: MLflow tracking output (local file backend)

---

## The data pipeline (Splunk → Prometheus → combined dataset)

### The mental model

Imagine you’re trying to answer this question continuously:

> “Is this service behaving like it usually does, right now?”

But instead of staring at dashboards, we teach a model what *normal* looks like from historical metrics. Then we ask it to flag windows that don’t look normal.

### How timestamps happen (important)

Right now, the pipeline is **event-driven** at collection time:

- Splunk provides a list of event timestamps (often irregular).
- Prometheus is queried at **those timestamps** using **instant queries**.

That means your raw dataset is not necessarily “a metric every minute”. It’s more like:

- “metrics sampled when something interesting was logged”.

This is great when your anomalies are strongly coupled to log events.
It’s weaker if you want continuous monitoring of slow drifts or silent failures.

### What gets produced

After combining, the dataset is a table with repeated timestamps (one row per service per timestamp) that includes:

- `timestamp` / `timestamp_nanoseconds`
- `service`
- metrics like `cpu`, `memory`, `latency`
- `severity_level` (derived from Splunk events)

This combined dataset is the raw material for the model.

---

## Preprocessing: turning tables into “something a model can learn from”

The model expects a dense 3D tensor shaped like:

- **time × service × feature** (written `[T, S, F]`)

But raw observability data rarely arrives like that.

### Two approaches you can switch between

The key question is: **What is “time” in the model?**

#### 1) `approach=event` (irregular timeline)

- Keep only the timestamps that actually exist in the combined dataset.
- Still densify across services so the model always sees all services at each timepoint.

Intuition:

- “Only look at moments that happened; don’t invent missing minutes.”

This makes the model’s time axis irregular. Windows represent “the next N observed moments”, not “the next N minutes”.

#### 2) `approach=grid` (fixed time grid)

- Snap timestamps onto a fixed grid (e.g. `1s`, `30s`, `1min`).
- Create every timestamp from min..max on that grid.
- Fill missing service values via forward-fill (then normalize, then fill remaining NaNs with 0).

Intuition:

- “Pretend we had a steady heartbeat, then fill in the gaps.”

This is *not* the same as collecting more metrics. It’s post-processing: you’re reshaping the same samples into a more regular tensor.

### How to pick a grid frequency

- If you choose a very coarse grid (like `1min`) on a very short dataset, many timestamps collapse together → you get too few points.
- If you choose a very fine grid (like `1s`) on a long time range, you can explode the number of timepoints.

A good way to think about it:

- Your grid should roughly match the “resolution” you want the model to notice.
- If you care about minute-level changes, `30s`–`1min` is reasonable.
- If your Splunk timestamps are microsecond-level and you want to preserve short bursts, `1s` or `5s` can be useful.

---

## The model: “learn normal by reconstructing it”

### What the model is doing, in one sentence

The model tries to **rebuild the input window**. If it can’t rebuild it well, that window is considered unusual.

Think of it like learning someone’s handwriting:

- After seeing lots of normal writing, you can reproduce it.
- If someone writes in a totally different style, your reproduction gets worse.

That “worse reproduction” becomes the anomaly score.

### Architecture (high-level)

The core model is a hybrid:

- an **autoencoder** idea (compress + reconstruct)
- with **attention/transformers** to model dependencies:
  - across services (who influences who)
  - across time (what patterns unfold in a window)

There are two attention mechanisms in play:

1. **Cross-service attention**: for each timestamp, the model can “look across services”
   - Intuition: “when checkout goes weird, auth often looks weird too”.

2. **Temporal transformer encoder**: for each service, the model can “look across time”
   - Intuition: “a spike that lasts 5 steps is different from a single blip”.

### What are “tokens” here?

In an LLM, tokens are words/subwords.
Here, you can think of:

- a **service** as an identity (like a word ID)
- plus its **observed metrics** (like the word content)

The model uses both:

- a learned service identity embedding (“this is service A vs service B”)
- a feature embedding (“this is what service A looks like right now”)

### Time features

Each window also has basic time-of-day/time-of-week signals (hour/minute/day/second normalized). Intuition:

- “Monday 9AM behaves differently than Sunday 3AM.”

---

## Training and inference (how anomalies are produced)

### Training

Training uses windows of length `window_size`.

- The model receives a window and tries to reconstruct it.
- Loss is reconstruction error (MSE).

A simple mental picture:

- You train on many “normal-ish” windows.
- The model becomes very good at reproducing those patterns.

### Thresholding

After training, you pick a threshold from validation data (e.g., 95th percentile).

- If a new window’s reconstruction error is above the threshold, it is marked anomalous.

This is like saying:

- “Only alert on the top 5% weirdest windows (relative to validation).”

### Inference

At inference time:

- the pipeline loads the model
- converts the latest dataset into `[T,S,F]`
- slides over windows
- produces an `error_score`

The repo includes a one-shot mode to run inference once and exit, and also a scheduler mode that runs periodically.

---

## Usage (practical)

### Start the normal service (scheduler/server)

- `./main.sh`

This schedules training and inference jobs for configured service sets.

### Run inference once and compare approaches

This runs the exact same inference twice (or more), once per variant, and saves separate outputs per variant:

- `./main.sh --run-once-inference --active-set transfer --inference-variants event,grid:1s`

This writes:

- `output/transfer/microservice_graph_with_results_event.png`
- `output/transfer/microservice_graph_with_results_grid_1s.png`
- and `output/transfer/inference_summary_*.json`

---

## Evaluation: how we compare approaches without waiting for real incidents

### Why synthetic anomalies exist

If we only evaluate on real production data, we often don’t know the ground truth:

- Was this actually an incident?
- Did the alert happen “in time”?

Synthetic anomalies solve this by creating a controlled experiment:

- Take a normal-ish dataset.
- Inject a known problem pattern.
- Check whether the model’s scores rise where we injected the problem.

### What gets injected (intuitive)

The injection utilities support:

- **spike**: “sudden burst” (e.g., latency jumps for 5 steps)
- **dropout**: “metric disappears / becomes zero”
- **level_shift**: “new normal” after a deploy (everything is a bit higher)
- **noise**: “shaky signal” (jitter)

These injections can be applied to one service + one feature for a known time range.

### The synthetic benchmark script

- `scripts/synthetic_benchmark.py`

It can compare variants in one run:

- `poetry run python scripts/synthetic_benchmark.py --variants event,grid:1s,grid:30s`

Outputs:

- a comparison table
- and optional JSON results (`--out output/synth_compare.json`)

### How to interpret AUC and F1 here

The benchmark evaluates **windows**.

- A window is labeled “anomalous” if it overlaps any injected anomaly.

Key metrics:

- **ROC AUC** (ranking quality)
  - Intuition: “If I pick one anomalous window and one normal window, how often does the anomalous one get a higher score?”
  - Great for comparing approaches because it doesn’t require choosing one specific threshold.

- **Precision / Recall / F1** (alerting quality)
  - Precision: “When I alert, how often am I right?”
  - Recall: “How many of the true anomalies did I catch?”
  - F1: balances the two.

Operationally:

- If you hate false alarms, prioritize **precision**.
- If you hate missed incidents, prioritize **recall**.

---

## Test data (what it is and what it is not)

### Unit test fixtures

`tests/resources/combined_dataset_test.json` is a small combined dataset used mainly for tests and quick dev checks.

Important caveats:

- It may not contain the same services as your configured service set.
  - That’s why you may see warnings like “missing expected services” or “unexpected services”.
- It is small enough that coarse grids like `grid:1min` can collapse it into too few timepoints.

This dataset is best thought of as:

- “a minimal example to ensure code paths work”

…not a realistic benchmark for production behavior.

### Output datasets

When you run the real pipeline you’ll see:

- `output/<active_set>/training_dataset.json`
- `output/<active_set>/inference_dataset.json`

These are closer to real usage and will usually be better for comparing approaches.

---

## A subtle but important distinction: preprocessing vs collection

It’s easy to assume:

- “grid approach” == “I collected a metric every second/minute”.

But right now:

- **collection** is still event-driven (Prometheus queried at Splunk event timestamps)
- **grid** is post-processing that *fills in a grid* after the fact

If your goal is truly continuous monitoring, the next technical step is to add a Prometheus **range query** mode (querying a time range with a fixed `step`).

---

## Where to look next

If you’re experimenting with approaches, these are the main knobs:

- Preprocessing:
  - `DATA_APPROACH=event|grid`
  - `TIME_GRID_FREQ=1s|30s|1min|...` (used by grid)
  - The `--inference-variants` flag for one-shot comparisons

- Evaluation:
  - `scripts/synthetic_benchmark.py --variants event,grid:1s,...`

And the practical workflow is:

1. Use synthetic benchmark to quickly see which preprocessing makes scores separate better.
2. Run one-shot inference for the top 2 variants and compare output graphs + anomaly rates.
3. Only then decide whether to invest in continuous Prometheus sampling.
