# Smart Traffic Signal Control System Using Computer Vision and Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![RL Library: Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)

## Abstract

This project presents an adaptive traffic signal control system for a 4-way intersection. A fine-tuned **DEtection TRansformer (DETR)** detects vehicles, **SORT** tracks them, per-lane zones turn the tracks into queue lengths, and a **Proximal Policy Optimization (PPO)** agent decides the signal phase. A **vision-language model (Claude)** reads the camera frame to confirm emergency vehicles (signal pre-emption) and incidents (reports for the operations room). The agent is trained in a fast queue simulation calibrated against the **SUMO** microscopic simulator, and is evaluated in both, on identical traffic, against fixed-time, vehicle-actuated and longest-queue-first control.

---

## 1. Introduction

Traffic congestion is a critical urban challenge leading to economic losses and increased emissions. Conventional traffic lights often rely on fixed timers or inductive loops, which do not adapt to real-time demand. This work proposes an end-to-end pipeline that:
1.  **Perceives** the intersection using computer vision.
2.  **Tracks** vehicles to estimate the demand on each approach.
3.  **Decides** the signal phase using a trained deep reinforcement learning agent.

---

## 2. System Architecture

### 2.1. Perception Module (Vision)
* **Model:** DETR (ResNet-50 backbone), fine-tuned on UA-DETRAC (Car, Bus, Van, Others).
* **Labels:** COCO JSON files keep the dataset ids (1-4); the model uses contiguous labels 0-3, because HF DETR reserves index `num_labels` for "no object".

### 2.2. State Estimation (Tracking + Queues)
* **Algorithm:** SORT (Kalman filter + Hungarian matching). The `DeepSortTracker` class name is historical; there is no appearance model.
* **Queues (`src/queue_estimator.py`):** one polygon per lane (8 lanes, `video.lane_zones` in the config, drawn with `calibrate_zones.py`). A vehicle counts as queued only when the ground point of its box moves slower than `video.stopped_speed_px_s`, so moving traffic is not mistaken for a queue.

### 2.3. Control Module (Reinforcement Learning)
* **Training recipe (`train_rl_agent.py`):**
  1. The policy first imitates the rule-based expert (actuated for cyclic, longest-queue for free) with behavior cloning + DAgger.
  2. PPO (Stable-Baselines3) fine-tunes it, one step per second, with `VecNormalize` on observations and rewards and 4 parallel environments.
  3. Every episode draws a traffic scenario (low, medium, rush hour, asymmetric main road).
  4. The best policy on validation seeds (relative to actuated control) is kept.
* **Environment (`src/environment.py`, V6):** 8 lanes, 4 phases, 1 step = 1 second, min green 5 s, max green 60 s, **3 s yellow after every switch**, and **2 s start-up lost time** at the beginning of each green.
* **Calibration against SUMO:** saturation flow 0.5 veh/s per lane (1800 veh/h, measured in SUMO), default demand 0.02–0.12 veh/s per lane.
* **Action modes:**
  * `cyclic` — `Discrete(2)`: extend the current phase or advance to the next one in the fixed order.
  * `free` — `Discrete(4)`: choose which phase gets green next.
* **Decision points:** during training the agent is only queried when its action can change the signal (not during yellow or minimum green).
* **Observation (31-dim):** queues (8), phase one-hot (4), green timer, next-phase demand, queue trend (8), waiting time per lane (8), yellow progress.
* **Reward (`reward_type`):** `queue` (default) = minus the total queue each second, i.e. total vehicle delay, which is exactly the evaluation metric; `shaped` = the original multi-term reward.
* **Baselines (`src/baselines.py`):** fixed time (30 s), vehicle-actuated (gap-out) and longest-queue-first. All policies are evaluated on the same seeds with identical random arrivals, and the worst lane waiting time is reported to check fairness.

### 2.4. Closed-Loop Evaluation in SUMO
`src/sumo_env.py` exposes the same interface as the fast environment, but the traffic is simulated vehicle by vehicle by SUMO: acceleration, braking, gaps and start-up delay. It uses an 8-lane intersection with protected left turns and yellow on the lanes that just lost green. Agents trained in the fast model are evaluated there without retraining (`compare_agents.py --sim sumo`). Unlike a recorded video, every signal decision changes what happens next.

### 2.5. Vision-Language Model (VLM)
`src/vlm.py` sends the camera frame (JPEG) and the sensor context (phase, queues, why the check was triggered) to Claude (`claude-opus-5-5`, low effort, JSON-schema structured output). It returns:
* **emergency vehicle:** present, approach, straight or left → `track_video.py` pre-empts the signal (green for that lane for `vlm.preempt_s` seconds). Only medium or high confidence triggers pre-emption.
* **incident:** type (collision, breakdown, blocked lane, signal fault, ...), approach, severity → shown on screen and logged to `outputs/results/vlm_events.jsonl`.
* an **Arabic report** for the operations room and a one-line **English summary** for the dashboard.

Requests run on a background thread, so the video loop never waits. Declined requests are retried on the recommended fallback model (server-side fallbacks). A check runs every `vlm.scan_interval_s` seconds, and whenever a lane's stopped vehicles do not move for `vlm.stall_trigger_s` seconds; each check is one API request, so choose the interval with cost in mind. Without an API key (`ANTHROPIC_API_KEY`, or `ant auth login`) the system falls back to template reports and no pre-emption.

### 2.6. Multi-Intersection Demo
`demo_system.py` connects two local agents to a central supervisor. It shows emergency-vehicle pre-emption with a green wave, and incident detection: a lane that gets green but does not discharge is reported to the operations room through `VLMReporter`. The report comes from Claude when credentials are available (text only, since the simulation has no camera), otherwise from the template.

---

## 3. Project Structure

```text
Smart-Traffic-System-Research/
├── configs/config.yaml     # Paths, dataset, model, training, RL, video zones and VLM settings
├── src/
│   ├── config.py           # Config loading, seeding, device, logger, label maps
│   ├── dataset.py          # UA-DETRAC XML -> COCO conversion, DETR dataset/loaders
│   ├── detector.py         # DETR wrapper (training + inference)
│   ├── trainer.py          # Detector training loop, mAP evaluation, early stopping
│   ├── inference.py        # Image / video detection and tracking pipelines
│   ├── tracker.py          # SORT tracker
│   ├── queue_estimator.py  # Tracks -> per-lane queues (lane polygons, stopped vehicles)
│   ├── viz.py              # Drawing helpers
│   ├── environment.py      # Gymnasium traffic-signal environment (V6)
│   ├── sumo_env.py         # Same environment on the SUMO microscopic simulator
│   ├── agents.py           # Training envs, imitation warm start, save/load with normalization
│   ├── baselines.py        # Fixed-time, actuated and longest-queue controllers
│   ├── vlm.py              # Claude vision-language scene analysis (+ template fallback)
│   ├── vlm_reporter.py     # Incident reports for the operations room
│   ├── intersection.py     # Local agent: RL control, emergency override, anomalies
│   └── supervisor.py       # Central supervisor: green wave for emergency vehicles
├── models/rl_agents/       # Agents trained in the fluid model (+ *_vecnormalize.pkl statistics)
├── models/rl_agents_sumo/  # Agents trained in SUMO (default for track_video.py)
├── experiments/            # Results and figures of the reported runs
├── tests/                  # pytest tests (environment, agents, VLM, SUMO, video controller)
├── train.py                # Train the DETR detector
├── validate_detection.py   # Per-class AP, overall mAP and latency
├── predict.py              # Run the detector on images
├── train_rl_agent.py       # Train cyclic and free PPO agents
├── compare_agents.py       # Agents vs baselines (--sim fluid | sumo), figures + results.json
├── validate_full.py        # Agents vs baselines across 4 traffic scenarios
├── test_rl_agent.py        # Terminal dashboard of one agent episode
├── calibrate_zones.py      # Draw the 8 lane polygons on your camera view
├── track_video.py          # Video -> DETR -> SORT -> queues -> RL (+ VLM pre-emption)
├── demo_system.py          # Multi-intersection architecture demo
└── requirements.txt
```

---

## 4. Installation

```bash
git clone https://github.com/abdulaziz1811/Smart-Traffic-System-Research.git
cd Smart-Traffic-System-Research
pip install -r requirements.txt
```

Python 3.10+ is required. A CUDA GPU or Apple Silicon (MPS) is recommended for the detector. SUMO (`eclipse-sumo`) and the Anthropic SDK are only needed for the SUMO evaluation and the VLM. For the VLM, set `ANTHROPIC_API_KEY` or run `ant auth login`.

---

## 5. Usage

### 5.1. Reinforcement Learning

```bash
python train_rl_agent.py --steps 600000                      # models/rl_agents/{cyclic,free}_agent
python compare_agents.py --seeds 10                          # fast queue model
python compare_agents.py --seeds 10 --sim sumo               # SUMO, closed loop
python train_rl_agent.py --sim sumo --steps 300000 --output models/rl_agents_sumo   # train in SUMO (~1 h)
python compare_agents.py --seeds 10 --sim sumo --agents-dir models/rl_agents_sumo
python validate_full.py --seeds 10                           # 4 traffic scenarios
python test_rl_agent.py --model models/rl_agents/free_agent
```

### 5.2. Vehicle Detection

```bash
python train.py                                  # fine-tune DETR (models/weights/)
python validate_detection.py --split test        # per-class AP + latency
python predict.py --image path/to/frame.jpg
```

### 5.3. Full System

```bash
python calibrate_zones.py --video path/to/traffic.mp4        # once per camera -> config video.lane_zones
python track_video.py --video path/to/traffic.mp4 --agent models/rl_agents_sumo/cyclic_agent
python track_video.py --video traffic.mp4 --no-vlm --no-display --output outputs/results/annotated.mp4
python demo_system.py
```

`track_video.py` queries the agent once per second of video with the same min/max green and yellow rules as the simulator. `tests/test_signal_controller.py` checks that it builds exactly the same observations. The video is pre-recorded, so the loop is open: the traffic in the video does not react to the signal. Use the SUMO evaluation for closed-loop results.

### 5.4. Tests

```bash
python -m pytest tests/
```

---

## 6. Results and Evaluation

### 6.1. Detection

| Metric | Value | Notes |
|---|---|---|
| mAP@50 (detection) | 92.4% | Reported from an earlier version of the code. Re-run `validate_detection.py`: the evaluation code previously crashed during post-processing, and the "Others" class could not be learned (see 2.1). |
| Inference speed | ~30 FPS (M2) | Not re-measured. `validate_detection.py` now synchronizes the GPU, and reports model forward time only. End-to-end FPS (decode + resize + tracking) is lower. |

### 6.2. Signal control (`experiments/exp4_mixed_demand/`, `experiments/exp5_sumo_training/`)

Test seeds 42–51, one hour of traffic each. The average queue is in vehicles per lane; lower is better. Agents in `models/rl_agents/` were trained in the fast queue model, agents in `models/rl_agents_sumo/` in SUMO.

| Strategy | Fluid model | SUMO (closed loop) | Max wait in SUMO (s) |
|---|---|---|---|
| Fixed timer 30 s | 6.49 | 5.81 | 102 |
| Actuated (gap-out) | 3.39 | 4.04 | 79 |
| Longest queue first | **3.27** | 3.93 | 146 |
| Cyclic agent, trained in fluid | 3.60 | 4.61 | 135 |
| Free agent, trained in fluid | 3.47 | **3.69** | 167 |
| Cyclic agent, trained in SUMO | 3.54 | 4.11 | 79 |
| Free agent, trained in SUMO | 3.35 | 3.83 | 157 |

* In **SUMO**, both free agents beat every rule-based controller. The best one, which never saw SUMO during training, has **36% less queue than the fixed timer**, 9% less than actuated control and 6% less than longest-queue-first.
* Training in SUMO fixed the cyclic agent's transfer gap (4.61 → 4.11): it now reproduces actuated control almost exactly. The SUMO-trained free agent is good in both simulators.
* In the fast queue model, the agents are within 2–10% of the rule-based controllers and about 45–48% better than the fixed timer.
* **Scenarios** (fluid model, `validate_full.py`): the agents stay at the level of the rule-based controllers in light traffic, rush hour and on an asymmetric main road. The fixed timer is 4x worse on the asymmetric road.

### 6.3. What the experiments showed (`experiments/exp3_*` to `experiments/exp5_*`)

1. **PPO from scratch never reached actuated control.** Imitation (behavior cloning + DAgger) of the rule-based expert reaches it. PPO fine-tuning added up to 1–3% on the training demand only.
2. **Discounting matters.** When one agent step spanned several seconds (decision steps) but was discounted as one step, fine-tuning made agents hold green too long. With one step per second it improved them.
3. **Training demand matters.** Agents trained on medium traffic only were the best controllers on that traffic, but collapsed at rush hour (queue 139 vs 53) and did not transfer to SUMO (6.56 vs 3.93). Training on mixed scenarios fixed both.
4. **Model selection is needed.** Longer PPO runs drifted away from good policies, most strongly on mixed demand. The best policy on validation seeds is kept, which on mixed demand was the imitation policy, both in the fluid model and in SUMO.
5. **Train where you evaluate.** Imitation learned in SUMO gave the best cyclic agent in SUMO and transfers back to the fluid model (54 min of training for both agents on 4 CPU cores).
6. For a single isolated intersection with good queue sensing, gap-out / longest-queue rules are already near-optimal. The remaining gains for RL are more likely with coordination between intersections, imperfect sensing, or objectives beyond delay.

**Validity note for earlier RL results.** Up to environment V4, switching phase had no cost: there was no yellow/all-red time. The trained agents learned to switch roughly every 6 s. On the same seeds, a *fixed timer with 6 s green* matched them (avg queue 2.27 vs 2.27 for the "Free" agent), and a simple actuated controller beat them (1.38). The "Cyclic" and "Free" agents were also trained on the identical `Discrete(2)` environment. The results in the `التجربه ...` folders were therefore produced with that flawed setup: experiment 1 used an even older 14-dim environment, and experiment 2 used V4.

---

## 7. Future Work

* Make PPO add value beyond imitation on mixed demand: a scale-free reward (delay relative to a reference controller per scenario), or a more conservative fine-tuning schedule.
* Multi-intersection coordination (MARL) on a SUMO corridor, where adaptive control has more to gain than at one isolated intersection.
* Calibrate the lane zones and queue threshold on real camera footage, and measure the VLM's emergency-vehicle recall and false-alarm rate on labelled clips.
* Deployment on edge devices (e.g. Jetson) and robustness to weather and night conditions.

---

## Author

Abdulaziz — Department of Artificial Intelligence, College of Computer Science and Engineering, University of Ha'il
