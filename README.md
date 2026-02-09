# Smart Traffic Signal Control System Using Computer Vision and Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![RL Library: Stable Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)

## Abstract

This project presents an adaptive, intelligent traffic signal control system designed to optimize intersection efficiency. By integrating a fine-tuned **DEtection TRansformer (DETR)** for real-time vehicle detection and a **Proximal Policy Optimization (PPO)** Reinforcement Learning agent for signal phase control, the system dynamically adjusts to fluctuating traffic demands. Experimental results demonstrate a significant reduction in average vehicle waiting times and queue lengths compared to traditional fixed-time control strategies.

---

## 1. Introduction

Traffic congestion is a critical urban challenge leading to economic losses and increased emissions. Conventional traffic light systems often rely on fixed timers or inductive loops, which fail to adapt to real-time traffic variance. This research proposes an end-to-end solution that:
1.  **Perceives** the environment using computer vision.
2.  **Tracks** vehicle flow to estimate queue density.
3.  **Decides** the optimal signal phase using a trained deep reinforcement learning agent.

---

## 2. System Architecture

The system consists of three modular components:

### 2.1. Perception Module (Vision)
* **Model:** DETR (ResNet-50 backbone).
* **Function:** Detects vehicles (Cars, Buses, Vans) in high-resolution video feeds.
* **Performance:** Achieved **92.4% mAP@50** on the UA-DETRAC dataset.

### 2.2. State Estimation (Tracking)
* **Algorithm:** SORT (Simple Online and Realtime Tracking).
* **Function:** Assigns unique IDs to detected vehicles to prevent double-counting and accurately estimate queue lengths per lane.

### 2.3. Control Module (Reinforcement Learning)
* **Algorithm:** Proximal Policy Optimization (PPO).
* **Environment:** A custom Gymnasium-based simulation of a 4-way intersection.
* **Reward Function:** Penalizes cumulative queue lengths and waiting times to ensure fairness and efficiency.
* **Outcome:** Improved traffic flow efficiency by approximately **40%** during simulation tests.

---

## 3. Project Structure

The repository is organized as follows:

```text
Smart-Traffic-System-Research/
├── configs/               # Configuration files for model and training parameters
├── src/                   # Source code modules
│   ├── detector.py        # Object detection model wrapper
│   ├── tracker.py         # SORT tracking implementation
│   ├── environment.py     # RL environment logic (Gymnasium)
│   ├── trainer.py         # Training loops and evaluation metrics
│   └── inference.py       # Inference pipelines
├── models/                # Pre-trained models and checkpoints
├── train.py               # Script for training the DETR detector
├── train_rl_agent.py      # Script for training the RL agent
├── demo_system.py         # Real-time system demonstration
└── requirements.txt       # Project dependencies
4. Installation and Setup
Prerequisites

Python 3.9 or higher

CUDA-enabled GPU (recommended) or Apple Silicon (MPS)

Installation Steps

Clone the Repository:

Bash
git clone [https://github.com/abdulaziz1811/Smart-Traffic-System-Research.git](https://github.com/abdulaziz1811/Smart-Traffic-System-Research.git)
cd Smart-Traffic-System-Research
Install Dependencies:

Bash
pip install -r requirements.txt
5. Usage
5.1. Reinforcement Learning Training

To train the PPO agent from scratch using the defined environment parameters:

Bash
python train_rl_agent.py
Parameters such as learning rate, total timesteps, and reward weights can be modified in configs/config.yaml.

5.2. Vehicle Detection and Tracking

To process a video file and generate a tracked output video:

Bash
python track_video.py --sequence <VIDEO_NAME> --output outputs/results/tracked.mp4
5.3. Full System Demonstration

To visualize the AI agent controlling the traffic signal based on video input:

Bash
python demo_system.py
6. Results and Evaluation
The system was evaluated using both detection metrics and traffic flow efficiency metrics.

Metric	Value	Description
mAP@50	92.4%	Mean Average Precision for vehicle detection.
Avg Reward	-6,860	Cumulative reward (higher is better, improved from baseline -11,900).
Inference Speed	~30 FPS	Real-time performance on M2 Silicon hardware.
7. Future Work
Integration with multi-intersection environments using Multi-Agent Reinforcement Learning (MARL).

Deployment on edge devices (e.g., Jetson Nano) for on-site testing.

incorporating weather conditions into the detection pipeline.

Author
Abdulaziz Department of Artificial Intelligence College of Computer Science and Engineering, University of Ha'il