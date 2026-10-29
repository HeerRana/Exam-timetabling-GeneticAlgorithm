# Constraint-Aware University Exam Timetabling Using Genetic Algorithms

[![Soft Computing](https://img.shields.io/badge/Course-Soft%20Computing-blue.svg)](https://github.com/HeerRana/-Exam-timetabling_usingGeneticAlgorithm)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](requirements.txt)
[![Status](https://img.shields.io/badge/Feasibility-0%20Hard%20Violations-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A complete Soft Computing course project implementing an evolutionary optimization framework for solving the **NP-hard University Examination Timetabling Problem (ETP)** under real-world hard and soft constraints.

---

## 📌 Problem Overview

Examination timetabling is a combinatorial optimization problem. In an academic department, scheduling exams across limited timeslots and rooms is subject to strict logistical and human constraints:
1. **Student Clash (Hard)**: No student can sit two exams scheduled at the same time.
2. **Room Clash (Hard)**: A single room cannot host multiple exams simultaneously.
3. **Room Capacity Overflow (Hard)**: Exam enrollment must not exceed the physical room seating capacity.
4. **Consecutive Paper Load (Soft)**: Minimize student fatigue caused by back-to-back papers on the same day (e.g., Morning and Afternoon sessions).

Deterministic approaches (such as greedy heuristics) frequently stagnate in suboptimal local minima. This project demonstrates how an evolutionary **Genetic Algorithm (GA)** navigates the combinatorial search space, avoids premature convergence, and reliably converges to a **0 hard-violation, clash-free schedule**.

---

## 🧬 Soft Computing & Evolutionary Model

### 1. Chromosome Representation
Each candidate solution (chromosome) is represented as a 1D discrete integer vector of length $N = 10$ (one gene per course):

$$\text{Gene}_i \in \{0, 1, \dots, (N_{\text{slots}} \times N_{\text{rooms}}) - 1\}$$

Each gene maps directly to a 2D timeslot-room assignment:
$$\text{Slot}_i = \lfloor \text{Gene}_i / N_{\text{rooms}} \rfloor, \quad \text{Room}_i = \text{Gene}_i \pmod{N_{\text{rooms}}}$$

### 2. Constraint Formulation

| Constraint Type | Description | Mathematical Formulation | Penalty Weight |
|---|---|---|---|
| **Hard (Student Clash)** | No student has two overlapping exams | $\sum_{i < j} \text{Overlap}_{ij} \cdot \mathbb{I}(\text{Slot}_i = \text{Slot}_j)$ | $w_{\text{hard}} = 100$ |
| **Hard (Room Clash)** | No room double-booking in the same slot | $\sum \mathbb{I}((\text{Slot}_i, \text{Room}_i) = (\text{Slot}_j, \text{Room}_j))$ | $w_{\text{hard}} = 100$ |
| **Hard (Room Overflow)** | Headcount must fit within room seating capacity | $\sum_{i} \mathbb{I}(\text{Enrolled}_i > \text{Capacity}_{\text{Room}_i})$ | $w_{\text{hard}} = 100$ |
| **Soft (Consecutive Load)** | Minimize fatigue from back-to-back slots (e.g. AM & PM) | $\sum_{i, j} \text{Overlap}_{ij} \cdot \mathbb{I}(\text{SameDayAdjacent}(\text{Slot}_i, \text{Slot}_j))$ | $w_{\text{soft}} = 1$ |

### 3. Fitness Function

$$\text{Fitness} = - \Big( 100 \times (\text{Student Clashes} + \text{Room Clashes} + \text{Capacity Overflows}) + \text{Consecutive Clashes} \Big)$$

- Feasible timetables require **Hard Violations = 0**.
- Higher fitness (closest to 0 or positive) corresponds to superior schedule quality.

### 4. Genetic Operators
- **Selection**: Tournament Selection ($k = 3$) to maintain steady selection pressure without premature convergence.
- **Crossover**: Single-Point Crossover with crossover probability $P_c = 0.85$.
- **Mutation**: Random Gene-Reset Mutation with probability $P_m = 0.12$.
- **Elitism**: Top 2 elite individuals are carried over unaltered to each generation to guarantee monotonic fitness progression.
- **Population Size**: 48 individuals over 80 generations.

---

### 5. How does the Genetic Algorithm improve the timetable?
The algorithm uses four key operations.

- **Step 1** — Initialize the population

Generate 48 candidate timetables. Initially, many may contain student clashes, room clashes, or capacity violations.



- **Step 2** — Evaluate and select

Calculate each timetable's fitness, then use tournament selection with \(k=3\): compare three candidates and select a better one as a parent.



- **Step 3** — Crossover

With crossover probability \(P_c=0.85\), combine portions of two parent chromosomes to create offspring. The project uses single-point crossover.



- **Step 4** — Mutation and elitism

Randomly change genes with mutation probability \(P_m=0.12\). Preserve the two best individuals unchanged in the next generation.



- **Step 5** — Repeat

Evaluate the new population and repeat for up to 80 generations, aiming for a feasible timetable with a low soft penalty.

---

## 📊 Experimental Results & Benchmark

The genetic algorithm was benchmarked against baseline strategies across identical problem instances (10 exams, 48 students, 6 slots across 3 days, 3 rooms with capacities of 32, 24, and 18):

| Optimization Strategy | Mean Hard Violations | Best Hard Violations | Feasible (0 Violations)? | Soft Consecutive Penalty |
|---|:---:|:---:|:---:|:---:|
| **Random Assignment** (40 runs) | $9.35 \pm 1.8$ | 6.00 | ❌ No | — |
| **Greedy Heuristic Packer** | 2.00 | 2.00 | ❌ No (Stuck in local optimum) | 14 |
| **Genetic Algorithm (Ours)** | **0.00** | **0.00** | ✅ **Yes (Optimal)** | **Minimized** |

### Key Observations:
- **Random Search** fails completely due to the dense student overlap between common core courses.
- **Greedy Allocation** sorts courses by largest enrollment and assigns slots myopically. While it solves capacity overflow, it gets trapped in slot conflicts when packing downstream courses.
- **The Genetic Algorithm** successfully balances exploration and exploitation, completely resolving all hard constraints within fewer than 60 generations.

---

## 📁 Repository Structure

```text
├── notebooks/
│   ├── exam_timetabling_ga.ipynb    # Main Jupyter Notebook (CPU-friendly, runnable in VS Code / Jupyter)
│   └── requirements.txt             # Dependencies specification
├── scripts/
│   └── verify_timetable.py          # Standalone Python CLI runner & sanity-checker
├── requirements.txt                 # Project-level Python dependencies
├── .gitignore                       # Python & Jupyter ignore rules
└── README.md                        # Documentation
```

---

## 🚀 Getting Started

The project runs on **standard CPU** (no GPU, CUDA, or external cloud platform required).

### 1. Clone the repository
```bash
git clone https://github.com/HeerRana/-Exam-timetabling_usingGeneticAlgorithm.git
cd -Exam-timetabling_usingGeneticAlgorithm
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project

#### Option A: Jupyter Notebook (Recommended)
Open VS Code or launch Jupyter Lab:
```bash
jupyter lab notebooks/exam_timetabling_ga.ipynb
```
Select the `.venv` kernel and run all cells. The notebook will:
- Generate the department enrollment matrix and course overlap graph.
- Execute baseline benchmarks (Random vs. Greedy vs. GA).
- Plot the fitness evolution curve across 80 generations.
- Render the final clash-free timetable grid.

#### Option B: Standalone CLI Script
Run directly from terminal:
```bash
python scripts/verify_timetable.py
```

---

## 🎓 Academic / Course Submission Guide

If submitting this for a Soft Computing course evaluation:

- **Problem Statement**: NP-hard examination scheduling subject to conflicting multi-stakeholder constraints.
- **Soft Computing Technique**: Evolutionary Computation — Genetic Algorithm with customized discrete encoding, tournament selection, single-point crossover, and elitism.
- **Novelty / Core Deliverable**: Comparative benchmark establishing empirical superiority of evolutionary metaheuristics over random search and greedy constructive heuristics.
- **Key Takeaway**: The GA consistently identifies feasible, zero-clash schedules while optimizing secondary student comfort metrics.

---

## 📄 License

This project is licensed under the MIT License.
