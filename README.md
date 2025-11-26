# 🏎️ Strategic Evolution: F1 Strategy Optimizer

**Strategic Evolution** is a simulation engine designed to solve the Formula 1 pit-stop strategy problem.
Developed as a capstone project for the **MSc in Economics and Data Science**, it contrasts **Heuristic Methods (Greedy)** with **Meta-Heuristic Optimization (Genetic Algorithms)** to solve a complex resource allocation problem under non-linear constraints.

## Project Overview

In Formula 1, tyres are a **scarce resource** that depreciates over time. The strategic problem is determining the optimal allocation of this resource over a finite time horizon (race laps) to minimize the total cost (race time).

This project simulates race dynamics using real telemetry data, introducing complex physical constraints that mimic the real-world trade-offs faced by race strategists:
* **Fuel Effect vs. Physical Degradation:** Disentangling the lap time gain from fuel burn vs. the loss from tyre wear.
* **Structural Fatigue ("The Cliff"):** Modeling non-linear performance drops using quadratic decay functions.
* **Thermodynamic Latency:** Simulating the "warm-up" phase where harder compounds lose time in the out-lap.
* **Logistical Constraints:** Dynamic Pit Loss calculation and Traffic (Dirty Air) penalties.


## Algorithmic Core

### 1. The Challenger: Genetic Algorithm (Evolutionary)
A meta-heuristic approach that mimics natural selection to find the Global Optimum.
* **Genome:** A strategy is represented as a sequence of stints.
* **Evolution:** Uses **Tournament Selection**, **Crossover** (mixing strategies), and **Adaptive Mutation** to explore the solution space.
* **Strength:** It can plan long-term, often sacrificing short-term speed (e.g., managing tyres) for a net strategic gain (e.g., avoiding an extra pit stop).

### 2. The Benchmark: Evaluative Greedy Algorithm (Heuristic)
An advanced "Smart Greedy" algorithm representing local optimization.
* **Logic:** Instead of following fixed rules, it runs a predictive simulation at every decision point to choose the compound with the best immediate performance for the next stint.
* **Weakness:** It suffers from **Strategic Myopia** (Short-termism). It often chooses the fastest tyre *now*, ignoring future structural limits, leading to suboptimal cumulative results (e.g., forced late pit stops).


## ⚙️ Technical Architecture

The project is structured into three modular components:

### `data_model.py` (ETL & Modeling)
* **Data Extraction:** Fetches real-time telemetry from the official F1 API via `FastF1`.
* **Cleaning:** Filters out non-representative laps (Safety Car, In/Out laps) to isolate pure race pace.
* **Regression Analysis:** Uses Linear Regression (`sklearn`) to estimate the base pace and degradation coefficients ($y = mx + q$) for each compound.
* **Dynamic Calibration:** Automatically calculates the specific Pit Loss for the chosen circuit using the median of historical pit stops.

### `optimizers.py` (The Simulation Engine)
Contains the physics engine and the algorithmic logic:
* **Physics Engine:**
    * `NON_LINEAR_WEAR`: Simulates exponential degradation ($t^2$) to punish over-extended stints.
    * `WARMUP_PENALTY`: Adds time loss for the first lap based on compound hardness.
    * `MAX_LIFE`: Enforces "Pirelli Limits" to prevent structural failure.
* **Classes:**
    * `GeneticOptimizer`: Implements the evolutionary loop (Pop Size: 80, Generations: 60).
    * `GreedySolver`: Implements the look-ahead heuristic logic.

### `main.py` (Orchestrator)
* **Interactive CLI:** Allows the user to select the Season and Grand Prix dynamically.
* **Visualization:** Generates a convergence plot comparing the Genetic evolution against the Greedy baseline.


## How to Run

### Prerequisites
* Python 3.8+
* `pip`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/f1-strategy-optimizer.git](https://github.com/YOUR_USERNAME/f1-strategy-optimizer.git)
    cd f1-strategy-optimizer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the simulation:**
    ```bash
    python main.py
    ```

### Usage Example
Follow the on-screen prompts:
1.  Enter Year: `2024`
2.  Select Race: `16` (Italian Grand Prix)
3.  Observe the strategic comparison in the terminal and the generated plot.


## Results & Analysis

The simulation consistently demonstrates the superiority of Global Optimization (Genetic) over Local Optimization (Greedy).

**Case Study: Chinese Grand Prix**
* **Greedy Strategy:** `HARD (50) -> HARD (4) -> MEDIUM (2)`
    * *Analysis:* The Greedy algorithm kept the fastest tyre (Hard) until the structural limit (Lap 50), forcing a double pit-stop sequence at the end to satisfy regulations.
* **Genetic Strategy:** `MEDIUM (13) -> HARD (43)`
    * *Analysis:* The Genetic algorithm accepted a slower start with Mediums to unlock a **One-Stop Strategy**, saving 23 seconds of pit loss and winning the race.

**Strategic Gain:** The Genetic Algorithm typically outperforms the Greedy baseline by **10-15 seconds** on complex circuits.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).