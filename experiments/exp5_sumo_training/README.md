# Experiment 5 — Training in SUMO (agents in `models/rl_agents_sumo/`)

Command:

```bash
python train_rl_agent.py --sim sumo --steps 300000 --output models/rl_agents_sumo
```

Same recipe as experiment 4, but every training, imitation and validation
episode runs in the SUMO microscopic simulator: DAgger warm start
(50k expert decisions + 2 rounds), 300k PPO steps, mixed demand (low /
medium / rush hour / asymmetric), and model selection on SUMO validation
seeds 1000–1007 scored as queue relative to actuated control. Wall time:
54 minutes for both agents on 4 CPU cores (`train_log.txt`).

Model selection again kept the imitation policies (step 0):

| Agent | Imitation accuracy | Warm start (x actuated) | PPO checkpoints (x actuated) |
|---|---|---|---|
| Cyclic | 97.8% | **1.006** | 1.20, 1.33, 1.15, 1.10, 1.05, 1.07 |
| Free | 98.0% | **0.969** | 1.29, 1.31, 1.26, 1.13, 1.13, 1.09 |

## Test seeds 42–51 (average queue per lane, 3600 s)

SUMO, closed loop (`sumo/`):

| Strategy | Avg queue | Max wait (s) |
|---|---|---|
| Fixed timer 30 s | 5.81 ± 1.34 | 102 |
| Actuated | 4.04 ± 1.19 | 79 |
| Longest queue | 3.93 ± 1.18 | 146 |
| Cyclic agent (trained in SUMO) | 4.11 ± 1.14 | 79 |
| Free agent (trained in SUMO) | **3.83 ± 1.21** | 157 |

Fluid model (`fluid/`), i.e. transfer in the other direction:

| Strategy | Avg queue | Max wait (s) |
|---|---|---|
| Fixed timer 30 s | 6.49 ± 1.51 | 102 |
| Actuated | 3.39 ± 1.02 | 136 |
| Longest queue | **3.27 ± 1.00** | 184 |
| Cyclic agent (trained in SUMO) | 3.54 ± 0.93 | 119 |
| Free agent (trained in SUMO) | 3.35 ± 0.97 | 199 |

## Comparison with the agents trained in the fluid model (experiment 4)

| Agent | Trained in | SUMO | Fluid |
|---|---|---|---|
| Cyclic | fluid (exp 4) | 4.61 | 3.60 |
| Cyclic | SUMO (exp 5) | **4.11** | **3.54** |
| Free | fluid (exp 4) | **3.69** | 3.47 |
| Free | SUMO (exp 5) | 3.83 | **3.35** |

* Training in SUMO fixed the cyclic agent's transfer gap (4.61 -> 4.11). It now
  reproduces actuated control almost exactly: same throughput, number of
  switches and worst wait.
* Both free agents beat every rule-based controller in SUMO. The
  SUMO-trained one is also the better of the two in the fluid model, and
  scores about the same averaged over both simulators (3.59 vs 3.58).
* PPO fine-tuning did not improve on the imitation policies in SUMO either.
  With mixed demand, heavy (rush-hour) episodes dominate the returns. Making
  RL add value beyond imitation likely needs a scale-free reward (e.g. delay
  relative to a reference controller per scenario) or a longer, more
  conservative fine-tuning schedule.
