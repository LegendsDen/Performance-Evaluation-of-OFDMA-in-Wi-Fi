
# 📊 OFDMA Service Discipline Simulator

This project evaluates and compares the performance of different **stateless service disciplines**—**FIFO Pooling**, **Max FIFO Pooling**, and **Max Pooling**—within Wi-Fi networks using **OFDMA** scheduling.

We focus specifically on **mean pooling size** (number of frames per jumbo frame) as a key performance metric under different scenarios:
- Increasing arrival rate to a fixed destination (Two-station scenario)
- Increasing number of destinations (Multi-destination scenario)

## 🎯 Objectives

- Simulate and evaluate **mean pooling size** across different Wi-Fi traffic scenarios.
- Analyze trade-offs between **throughput**, **fairness**, and **system capacity**.
- Study how scheduling disciplines scale with:
  - Increasing **arrival rate** to a destination (2-station scenario)
  - Increasing **number of destinations** (multi-destination scenario)

---

## ⚙️ Implemented Disciplines

The simulator models and compares the following **stateless service disciplines**:

| Discipline         | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **FIFO Pooling**   | Pure FIFO; aggregates same-destination frames or uses OFDMA for different ones. |
| **Max FIFO Pooling** | Starts from the FIFO head; selects the combination that maximizes pooling.    |
| **Max Pooling**     | Greedy approach that selects frames (ignoring arrival order) to maximize jumbo frame size. |

---

## 🧪 Methodology

- **Language**: C++
- **Simulation Goals**:
  - Compute metrics like pooling size, sojourn time, load, unfairness
  - Analyze both **perfect** and **imperfect** OFDMA behavior
- **Scenarios Simulated**:
  - **Two-station**: One fixed arrival rate, one increasing
  - **Multi-destination**: Equal rate, but varying number of destinations

---

## 📈 Sample Results

### 📌 Mean Pooling Size vs Arrival Rate (Two-Station)
- Pooling size increases **non-linearly** as the system approaches saturation.
- **Max Pooling** shows highest efficiency.

### 📌 Mean Pooling Size vs Destination Count (Multi-Destination)
- All disciplines improve pooling with more destinations.
- **FIFO Pooling** saturates ~24 destinations.
- **Max FIFO Pooling** scales better and offers a good fairness-capacity trade-off.

---


## 🔧 Setup & Execution

### 🛠 Requirements
- Python 3.x (optional, for visualization)
- Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`

### ⚙️ Build
```bash
python TwoStation.py
```
---

## 👨‍💻 Author

Sushant Kumar
Tanay Goenka
Tanmay Mittal
Priyanshu Pratya
Course: CS348 – Performances Modelling Of Communication And Computer Systems 
Institution: IIT Guwahati  
Year: 2025



