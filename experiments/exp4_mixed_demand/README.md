# Experiment 4 — Mixed-demand training (final agents in `models/rl_agents/`)

Setup: fluid model V6 calibrated to SUMO (0.5 veh/s per lane, 2 s start-up loss,
3 s yellow), reward `queue`. Every training episode draws a scenario: low
(0.01–0.03), medium (0.02–0.12), rush hour (0.10–0.17) or asymmetric main road
(N/S busy, E/W light). DAgger warm start (50k expert decisions + 2 rounds) +
600k PPO steps, with model selection on validation seeds 1000–1007 scored as
queue relative to actuated control (`train_log.txt`).

Model selection kept the imitation policies (step 0) for both agents: PPO
fine-tuning on mixed demand made the validation score worse (cyclic 1.05 -> up
to 1.21 x actuated, free 1.00 -> up to 1.29 x actuated). The final agents are
therefore DAgger imitation policies of actuated (cyclic) and longest-queue
(free) control, trained on the mixed demand.

Test seeds 42–51, 3600 s per episode.

## Fluid model, config demand (`fluid/`)

| Strategy | Avg queue | Max wait (s) | Served |
|---|---|---|---|
| Fixed timer 30 s | 6.49 ± 1.51 | 102 | 2106 |
| Actuated | 3.39 ± 1.02 | 136 | 2150 |
| Longest queue | **3.27 ± 1.00** | 184 | 2151 |
| Cyclic agent | 3.60 ± 1.04 | 131 | 2149 |
| Free agent | 3.47 ± 1.10 | 196 | 2152 |

## SUMO, closed loop, agents never trained in SUMO (`sumo/`)

| Strategy | Avg queue | Max wait (s) | Served |
|---|---|---|---|
| Fixed timer 30 s | 5.81 ± 1.34 | 102 | 2086 |
| Actuated | 4.04 ± 1.19 | 79 | 2117 |
| Longest queue | 3.93 ± 1.18 | 146 | 2119 |
| Cyclic agent | 4.61 ± 1.33 | 135 | 2114 |
| Free agent | **3.69 ± 1.31** | 167 | 2122 |

The free agent is the best controller in SUMO: 36% less queue than the fixed
timer, 9% less than actuated and 6% less than longest-queue control. In
experiment 3 (medium demand only) the same kind of agent scored 6.56 ± 4.35
in SUMO; mixed-demand training fixed the transfer.

## Scenarios, fluid model (`scenarios_*`), average queue

| Scenario | Fixed 30 s | Actuated | Longest queue | Cyclic agent | Free agent |
|---|---|---|---|---|---|
| Low | 0.92 | 0.36 | 0.27 | 0.36 | 0.29 |
| Medium | 4.63 | 3.05 | 2.96 | 3.16 | 3.16 |
| Rush hour | 64.01 | 52.55 | 52.85 | 52.99 | 52.69 |
| Asymmetric | 17.81 | 4.01 | 3.87 | 4.45 | 4.30 |

Compared with experiment 3, rush hour no longer breaks the free agent
(139.39 -> 52.69); on medium traffic it gives up a few percent (2.90 -> 3.16).
