# Final Year Project

This repository contains the code and experimental results for a Final Year Project on **robotic workcell layout optimization** using a hybrid **Genetic Algorithm (GA)** and **Reinforcement Learning (RL)** framework.

The objective is to optimize machine placement and orientation within a constrained workspace to **minimize robot travel distance** while ensuring **collision-free layouts**.

---

## 📁 Project Structure

- `algo/`  
  Core optimization algorithms, including:
  - Genetic Algorithm (GA)
  - RL-assisted GA (RL-GA)
  - Q-learning agent for adaptive parameter control  

- `func/`  
  Supporting utilities:
  - Machine and workspace data structures :contentReference[oaicite:0]{index=0}  
  - Collision detection and geometric feasibility checks :contentReference[oaicite:1]{index=1}  

- `visual/`  
  Visualization tools for:
  - Layout rendering  
  - Optimization progress (fitness, convergence) :contentReference[oaicite:2]{index=2}  

- `tests/`  
  Scripts and notebooks for benchmarking different problem instances and configurations  

- `results_*/`  
  Experimental outputs including:
  - Fitness convergence plots  
  - Layout evolution images  
  - Performance comparison data  

- `rl_agent/`  
  Stored Q-tables from trained RL agents  

---

## ⚙️ Requirements

- Python **3.8+**

### Required Packages

```bash
pip install numpy matplotlib
