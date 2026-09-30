# Smart Traffic Signal Control System Using Computer Vision and Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![RL Library: Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)

## Abstract

This project presents an adaptive traffic signal control system for a 4-way intersection. A fine-tuned **DEtection TRansformer (DETR)** detects vehicles, **SORT** tracks them to estimate per-approach demand, and a **Proximal Policy Optimization (PPO)** agent decides the signal phase. The agent is trained in a custom Gymnasium simulation and compared on identical traffic against a fixed-time plan and a vehicle-actuated controller.

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

### 2.2. State Estimation (Tracking)
* **Algorithm:** SORT (Kalman filter + Hungarian matching). The `DeepSortTracker` class name is historical; there is no appearance model.
* **Demand estimate:** tracked vehicles are counted in one zone per approach (`track_video.py`). The straight / left-turn split is currently an assumed 80/20 share.

### 2.3. Control Module (Reinforcement Learning)
* **Algorithm:** PPO (Stable-Baselines3).
* **Environment (`src/environment.py`, V5):** 8 lanes, 4 phases, 1 step = 1 second, min green 5 s, max green 60 s, and **3 s yellow/all-red lost time after every switch** (no lane is served).
* **Action modes:**
  * `cyclic` — `Discrete(2)`: extend the current phase or advance to the next one in the fixed order.
  * `free` — `Discrete(4)`: choose which phase gets green next.
* **Observation (22-dim):** queues (8), phase one-hot (4), green timer, next-phase demand, queue trend (8).
* **Reward:** quadratic queue cost + accumulated waiting cost + shaping terms for wasted green and premature switches + a bonus per vehicle served.
* **Baselines (`src/baselines.py`):** fixed time (30 s), vehicle-actuated (gap-out) and longest-queue-first. All policies are evaluated on the same seeds with identical random arrivals.

### 2.4. Multi-Intersection Demo
`demo_system.py` connects two local agents to a central supervisor. It shows emergency-vehicle pre-emption with a green wave, and incident detection: a lane that gets green but does not discharge is reported to the operations room by the (template-based) `VLMReporter`.

---

## 3. Project Structure

```text
Smart-Traffic-System-Research/
├── configs/config.yaml     # All paths, dataset, model, training and RL parameters
├── src/
│   ├── config.py           # Config loading, seeding, device, logger, label maps
│   ├── dataset.py          # UA-DETRAC XML -> COCO conversion, DETR dataset/loaders
│   ├── detector.py         # DETR wrapper (training + inference)
│   ├── trainer.py          # Detector training loop, mAP evaluation, early stopping
│   ├── inference.py        # Image / video detection and tracking pipelines
│   ├── tracker.py          # SORT tracker
│   ├── viz.py              # Drawing helpers
│   ├── environment.py      # Gymnasium traffic-signal environment (V5)
│   ├── baselines.py        # Fixed-time, actuated and longest-queue controllers
│   ├── intersection.py     # Local agent: RL control, emergency override, anomalies
│   ├── supervisor.py       # Central supervisor: green wave for emergency vehicles
│   └── vlm_reporter.py     # Template-based incident report generator
├── tests/                  # pytest sanity tests for the environment and agents
├── train.py                # Train the DETR detector
├── validate_detection.py   # Per-class AP, overall mAP and latency
├── predict.py              # Run the detector on images
├── train_rl_agent.py       # Train cyclic and free PPO agents (with curriculum)
├── compare_agents.py       # Benchmark agents vs baselines (figures + results.json)
├── validate_full.py        # Agents vs baselines across 4 traffic scenarios
├── test_rl_agent.py        # Terminal dashboard of one agent episode
├── track_video.py          # Video -> DETR -> SORT -> RL decision (real-time demo)
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

Python 3.9+ is required. A CUDA GPU or Apple Silicon (MPS) is recommended for the detector.

---

## 5. Usage

### 5.1. Reinforcement Learning

```bash
python train_rl_agent.py --steps 500000          # trains models/rl_agents/{cyclic,free}_agent.zip
python compare_agents.py --seeds 10              # figures + results.json in outputs/comparison/
python validate_full.py --seeds 10               # 4 traffic scenarios, outputs/validation/
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
python track_video.py --video path/to/traffic.mp4 --agent models/rl_agents/cyclic_agent
python demo_system.py
```

`track_video.py` queries the agent once per second of video with the same min/max green and yellow rules as the simulator. The video is pre-recorded, so the loop is open: the traffic in the video does not react to the signal.

### 5.4. Tests

```bash
python -m pytest tests/
```

---

## 6. Results and Evaluation

| Metric | Value | Notes |
|---|---|---|
| mAP@50 (detection) | 92.4% | Reported from an earlier version of the code. Re-run `validate_detection.py`: the evaluation code previously crashed during post-processing, and the "Others" class could not be learned (see 2.1). |
| Inference speed | ~30 FPS (M2) | Not re-measured. `validate_detection.py` now synchronizes the GPU, and reports model forward time only. End-to-end FPS (decode + resize + tracking) is lower. |
| RL control | to be regenerated | See the validity note below. |

**Validity note for earlier RL results.** Up to environment V4, switching phase had no cost: there was no yellow/all-red time. The trained agents learned to switch roughly every 6 s. On the same seeds, a *fixed timer with 6 s green* matched them (avg queue 2.27 vs 2.27 for the "Free" agent), and a simple actuated controller beat them (1.38). The "Cyclic" and "Free" agents were also trained on the identical `Discrete(2)` environment. The results in the `التجربه ...` folders were therefore produced with that flawed setup: experiment 1 used an even older 14-dim environment, and experiment 2 used V4. Retrain with the current environment and report results against the actuated baseline as well as the fixed timer.

---

## 7. Future Work

* Per-lane detection zones (instead of the assumed 80/20 split), and counting only stopped vehicles as queue.
* Closed-loop evaluation in a microscopic simulator (e.g. SUMO) calibrated with detected counts.
* Observation normalization (`VecNormalize`), and including accumulated waiting time in the observation (it is part of the reward).
* Multi-intersection coordination with Multi-Agent RL (MARL).
* Deployment on edge devices (e.g. Jetson) and robustness to weather and night conditions.

---

## Author

Abdulaziz — Department of Artificial Intelligence, College of Computer Science and Engineering, University of Ha'il
