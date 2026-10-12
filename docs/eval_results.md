# Evaluation Results

Reference laptop: Ryzen 5 3550H, 8 GB RAM, GTX 1650 (CPU-only inference), Windows.
Fill in the tables below from real trials. 10 trials per scenario per §14.

## Scenario Evaluation

| # | Scenario | Expected | Actual (fill in) | Trials (x/10) | Notes |
|---|----------|----------|-------------------|----------------|-------|
| 1 | Friend behind you at about 1 m, looking at screen | ALERT | | /10 | |
| 2 | Friend walking past, not looking | No alert | | /10 | |
| 3 | Friend far away (3 m+), looking | No alert | | /10 | |
| 4 | Friend at side angle (about 45 deg), looking | Report result (known weak case) | | /10 | |
| 5 | Trusted friend behind you | No alert | | /10 | |
| 6 | Two peekers | ALERT, one event logged per episode | | /10 | |
| 7 | You turn your own head | No alert | | /10 | |
| 8 | Low light, backlight | Report result | | /10 | |
| 9 | Peeker leaves | Overlay clears after about 2 s | | /10 | |
| 10 | Owner leaves the desk | Screen hidden after about 5 s (AWAY) | | /10 | |
| 11 | Owner returns, or a stranger/trusted person sits down | Only the owner's face clears AWAY | | /10 | |

## Metrics

| Metric | Target (§12) | Measured |
|--------|---------------|----------|
| FPS (640x480) | 15 or higher | 15-17 (observed 2026-10-12, stable after fixing the FPS measurement bug — see PROJECT.md Entry 12) |
| Alert latency (peeker confirmed to blur visible) | 0.7 s or less | (fill in — time a few trials with a stopwatch/phone) |
| Clear delay after peeker leaves | about 2 s | (fill in) |
| RAM | under 1 GB | (fill in — check Task Manager while running) |
| CPU | under 50% average | (fill in — check Task Manager while running) |
| False alerts per 10 min normal use (full system) | report | (fill in — run scripts/eval_naive_baseline.py --mode full for 10 min, no deliberate peeking) |
| False alerts per 10 min normal use (naive baseline) | report | (fill in — run scripts/eval_naive_baseline.py --mode naive for 10 min, no deliberate peeking) |

## Baseline Comparison ("2 faces means blur")

See `scripts/eval_naive_baseline.py`. Run both modes for the same 10-minute session
(you working normally, occasionally a second person walking past/sitting nearby but
NOT deliberately peeking) and compare how many false "episodes" each triggers.

| Run | Episodes triggered | Notes |
|-----|---------------------|-------|
| Naive baseline (face count >= 2) | | |
| Full system (distance + gaze filtering) | | |

## Observations

(Free-form notes: anything unexpected, lighting conditions, camera used, etc.)
