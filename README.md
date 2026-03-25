# AI-powered-5G-network-scheduler
AI-powered 5G network scheduler using PPO (Reinforcement Learning) with NVIDIA GPU acceleration and real-time visualization dashboard.
# 📡 AI-Powered 5G Scheduler (PPO + NVIDIA GPU)

An advanced AI-driven 5G Radio Access Network (RAN) scheduler built using Reinforcement Learning (PPO) and optimized with NVIDIA GPU acceleration.

This project simulates real-world 5G network conditions and dynamically allocates resources using an intelligent AI agent.

---

## 🚀 Features

- 🧠 Proximal Policy Optimization (PPO) based scheduler
- ⚡ GPU acceleration using CUDA (PyTorch + AMP)
- 📊 Real-time training visualization dashboard (Streamlit)
- 📡 Simulated 5G environment (SNR, CQI, traffic queues)
- 📁 Clean modular architecture (agents, env, utils, logs)
- 💾 Model checkpointing and logging system

---

## 🧠 How It Works

1. Simulates a 5G network environment with multiple users
2. AI agent observes:
   - Signal strength (SNR)
   - Channel quality (CQI)
   - Queue length (traffic demand)
3. PPO model learns optimal scheduling policy
4. Allocates resources dynamically to maximize throughput & efficiency

---

## 🏗️ Project Structure
5g_ai_scheduler/
├── agents/ # PPO RL agent
├── sim/ # 5G environment simulator
├── utils/ # preprocessing & logging
├── models/ # saved models
├── logs/ # training logs
├── checkpoints/ # trained weights
├── dashboard.py # real-time visualization
├── train.py # training pipeline
├── infer.py # inference script


---

## 📊 Results

- Stable reward convergence using PPO
- Improved scheduling efficiency over random allocation
- Real-time visualization of training progress

---

## ▶️ How to Run

### 1️⃣ Train Model
```bash
python3 train.py

python3 -m streamlit run dashboard.py
