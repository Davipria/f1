🏎️ Strategic Evolution: F1 Strategy Optimizer (Research Edition)
Strategic Evolution is a comprehensive simulation framework designed to solve the Formula 1 pit-stop strategy problem.

Originally developed as an MSc Economics and Data Science Capstone, this project contrasts Heuristic Methods (Greedy) with Meta-Heuristic Optimization (Genetic Algorithms) using real-world telemetry data. It now includes a full suite of statistical tools to validate algorithmic superiority across entire seasons.

🚀 Key Features
Real-World Telemetry: Powered by FastF1, extracting live timing data, tyre degradation, and pit loss metrics for any race from 2018–2024.

Advanced Physics Engine: Models non-linear degradation ("The Cliff"), fuel-corrected lap times, thermal warm-up penalties, and traffic (dirty air).

Dual-Algorithm Comparison:

Genetic Algorithm (GA): A global optimizer capable of sacrificing short-term pace for long-term strategic gain.

Smart Greedy: A predictive heuristic that simulates "myopic" decision-making (optimizing only for the next stint).

Scientific Validation Suite:

Batch Testing: Automates simulation across whole seasons (e.g., 22 races) with multiple seeds.

Statistical Rigor: Performs T-Tests, Cohen's d effect size analysis, and Win-Rate binomial tests.

Publication-Ready Plots: Generates LaTeX tables and Seaborn-style figures for academic papers.

📂 Project Structure
Plaintext
f1-main/
├── main.py                     # Interactive CLI for single-race simulation
├── data_model.py               # ETL: Fetches FastF1 data & models tyre curves (Regressions)
├── optimizers.py               # Core Logic: Physics Engine, Genetic Algo, & Greedy Solver
├── visualization.py            # Matplotlib visualizations (Gantt charts, Convergence)
├── config.py                   # Global physics constants & hyper-parameters
├── requirements.txt            # Dependencies
│
└── report/                     # [NEW] Research & Validation Module
    ├── batch_test.py           # Runs simulations on entire seasons automatically
    ├── statistical_analysis.py # Calculates p-values, CI, and descriptive stats
    └── generate_paper_plots.py # Generates thesis-quality figures from batch results
🛠️ Installation
Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/f1-strategy-optimizer.git
cd f1-strategy-optimizer
Install dependencies:

Bash
pip install -r requirements.txt
💻 Usage
1. Interactive Mode (Single Race)

Best for visualizing a specific Grand Prix strategy in detail.

Bash
python main.py
Workflow: Select a Year -> Select a Circuit.

Output: Prints the degradation coefficients, compares strategies lap-by-lap, and displays a Gantt chart comparison.

2. Research Mode (Batch Testing)

Best for validating the algorithm's performance over a full season.

Step A: Run the Batch Simulation Run the GA vs. Greedy comparison on every race of a specific year.

Bash
# Run 2024 season, 5 runs per circuit for statistical robustness
python report/batch_test.py --year 2024 --runs 5
Step B: Statistical Analysis Analyze the results JSON to check for statistical significance (p-values).

Bash
python report/statistical_analysis.py report/results/batch_results_2024_XXXX.json
Step C: Generate Thesis Plots Create publication-ready visualizations (Bar charts, Scatter plots, LaTeX tables).

Bash
python report/generate_paper_plots.py report/results/batch_results_2024_XXXX.json
🧠 Algorithmic Logic
The Physics Model

The simulator disentangles Fuel Effect (linear time gain) from Tyre Degradation (non-linear time loss).

Formula: Time(t)=BasePace+(Lin⋅t)+(Quad⋅t 
2
 )+WarmUp+Traffic

Constraints: Includes the "Pirelli Limit" (structural failure) and the "2-Compound Rule."

The Contenders

Feature	Genetic Algorithm (The Challenger)	Greedy Algorithm (The Benchmark)
Type	Meta-Heuristic (Stochastic)	Heuristic (Deterministic)
Horizon	Global (Full Race)	Local (Next Stint Prediction)
Strengths	Can execute "Undercuts" and "One-Stoppers" by sacrificing early pace.	extremely fast computation; guarantees locally optimal stints.
Weaknesses	Computationally expensive; requires tuning.	Suffers from "Strategic Myopia" (Short-termism).
📊 Sample Results
Based on 2024 Season simulations:

Average Improvement: The GA improves upon the Greedy strategy by an average of ~9.5 seconds per race.

Win Rate: The GA finds a better or equal strategy in 91% of simulations.

Statistical Significance: p<0.001 (One-sample t-test), confirming the results are not due to random chance.

📄 License
This project is open-source and available under the MIT License.

Data Source: All telemetry data is provided by the FastF1 library.