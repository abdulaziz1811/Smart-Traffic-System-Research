# Experiment 3 — V6 environment, imitation + PPO, medium demand only

Setup: fluid model V6 calibrated to SUMO (0.5 veh/s per lane, 2 s start-up loss, 3 s yellow),
reward `queue`, training demand 0.02–0.12 veh/s per lane (config range only).
Agents: DAgger warm start (50k expert decisions + 2 rounds) + 600k PPO steps,
best policy selected on validation seeds 1000–1004 (`train_log.txt`).
Test seeds 42–51, 3600 s per episode.

## Ablation (300k PPO steps, fluid model, test seeds) — `ablation.json`

| Method | Mode | Avg queue | Max wait (s) |
|---|---|---|---|
| Actuated (rule) | cyclic | 3.39 | 136 |
| Longest-queue (rule) | free | 3.27 | 184 |
| PPO from scratch, decision steps | free | 6.33 | 204 |
| Imitation only | cyclic | 4.06 | 162 |
| Imitation only | free | 3.82 | 191 |
| Imitation + PPO, decision steps | cyclic | 3.43 | 121 |
| Imitation + PPO, decision steps | free | 4.12 | 208 |
| Imitation + cautious PPO, decision steps | free | 5.06 | 274 |
| DAgger only | free | 3.30 | 177 |
| DAgger + PPO, per-second steps | cyclic | 3.35 | 130 |
| DAgger + PPO, per-second steps | free | **3.24** | 167 |

Findings: PPO from scratch never reached the rule-based controllers; DAgger
imitation reaches them; PPO fine-tuning with a step per second adds a little.
With decision steps (one step spanning several seconds, discounted per step)
PPO fine-tuning pushed agents to hold green too long.

## Final agents (`agents/`), test seeds 42–51

Fluid model (`fluid/`):

| Strategy | Avg queue | Max wait (s) |
|---|---|---|
| Fixed timer 30 s | 6.49 | 102 |
| Actuated | 3.39 | 136 |
| Longest queue | 3.27 | 184 |
| Cyclic agent | 3.46 | 135 |
| Free agent | 3.27 | 173 |

SUMO, closed loop, no retraining (`sumo/`):

| Strategy | Avg queue | Max wait (s) |
|---|---|---|
| Fixed timer 30 s | 5.81 | 102 |
| Actuated | 4.04 | 79 |
| Longest queue | 3.93 | 146 |
| Cyclic agent | 5.27 | 162 |
| Free agent | 6.56 ± 4.35 | 138 |

Scenarios, fluid model (`scenarios_*`), average queue:

| Scenario | Fixed 30 s | Actuated | Longest queue | Cyclic agent | Free agent |
|---|---|---|---|---|---|
| Low | 0.92 | 0.36 | 0.27 | 0.36 | 0.33 |
| Medium | 4.63 | 3.05 | 2.96 | 3.04 | **2.90** |
| Rush hour | 64.01 | 52.55 | 52.85 | 53.21 | 139.39 |
| Asymmetric | 17.81 | 4.01 | 3.87 | 6.22 | **3.85** |

Conclusions that led to the next experiment:
1. The free agent is the best controller on the traffic it was trained on,
   but collapses at rush hour (outside its training demand) -> train on
   mixed scenarios.
2. The rule-based controllers transfer to SUMO, the agents do not
   (sim-to-sim gap) -> train and validate in SUMO.
