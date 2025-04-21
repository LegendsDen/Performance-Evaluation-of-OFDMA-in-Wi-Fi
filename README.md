
# 📊 OFDMA Service Discipline Simulator

This project evaluates and compares the performance of different **stateless service disciplines**—**FIFO Pooling**, **Max FIFO Pooling**, and **Max Pooling**—within Wi-Fi networks using **OFDMA** scheduling.

We focus specifically on **mean pooling size** (number of frames per jumbo frame) as a key performance metric under different scenarios:
- Increasing arrival rate to a fixed destination (Two-station scenario)
- Increasing number of destinations (Multi-destination scenario)

## 📌 Objectives

- Analyze and simulate the **mean pooling size** under different service disciplines.
- Understand how system saturation impacts the ability to pool frames efficiently.
- Visualize the scalability of pooling mechanisms with arrival load and destination count.

## 🧪 Methodology

- Developed a **C++ simulator** that models Wi-Fi packet scheduling with different pooling disciplines.
- Measured **mean pooling size** by aggregating frames under varying:
  - Arrival rates (\( \lambda_2 \)) for a two-destination setup
  - Number of destinations (from 2 to 40) for scalability analysis
- Generated plots to visualize system behavior and performance bottlenecks.

---

## 📊 Key Results

### 📌 Mean Pooling Size vs Arrival Rate (Two-Station Scenario)
- Pooling size increases **exponentially** as arrival rate increases and system approaches saturation.
- **Max Pooling** performs slightly better than **Max FIFO** and **FIFO Pooling** near saturation.

### 📌 Mean Pooling Size vs Number of Destinations
- All disciplines show increased pooling size as destinations increase.
- **FIFO Pooling** hits its limit at 24 destinations.
- **Max FIFO Pooling** and **Max Pooling** continue scaling beyond that, with **Max FIFO Pooling** reaching the **highest pooling size** (~41 frames).

---

## 🔧 Setup and Execution

### Requirements
- C++ compiler (g++ or clang++)
- Python 3.x (for plotting, optional)
- Libraries (optional):
  - `matplotlib`
  - `numpy`
  - `pandas`