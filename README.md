# 🧬 Stem Cell Differentiation with PINNs and Numerical ODE Solvers

This project compares **Physics-Informed Neural Networks (PINNs)** and classical **ODE solvers** for modeling stem cell differentiation through the mutual inhibition of transcription factors **GATA-1** and **PU.1**. We used a nonlinear biological model adapted from [Schiesser, 2014] and applied both numerical and machine learning methods to solve and analyze its behavior under different biological scenarios.

---

## 🔍 Problem Overview

- A system of nonlinear ODEs describes the gene regulatory interaction between **PU.1** and **GATA-1**, two transcription factors critical to hematopoietic lineage decisions.
- We implemented multiple solution strategies including:
  - Classical solvers: **LSODE (R)**, **LSODA (Python)**, **Trapezoidal**, and **Radau**
  - Deep learning-based solver: **PINNs** implemented in **PyTorch**
- We compared performance under two different parameter configurations to assess solver accuracy, efficiency, and robustness.

---

## 📖 Contents

- 📄 [Extended Report](report/extended_report.md)
- 🧮 [Numerical Solver Notebook](notebooks/numerical_solution.ipynb)
- 🤖 [PINN Solver Notebook](notebooks/pinn_solution.ipynb)
- 📊 [Results & Visualizations](results/)

---

## 📚 Literature Review

PINNs combine data-driven learning with physical laws by embedding differential equations into the loss function. Literature shows their promise in modeling biological dynamics, especially with sparse or noisy data. However, they often struggle with stiffness or highly nonlinear systems, where traditional solvers like **Radau** or **LSODA** retain strong performance.

Full review [here](report/extended_report.md#literature-review).

---

## 🧠 ODE Model

$$ \frac{d[G]}{dt} = a_1 \frac{[G]^n}{\theta_{a1}^n + [G]^n} + b_1 \frac{\theta_{b1}^m}{\theta_{b1}^m + [G]^m[P]^m} - k_1[G] $$

$$ \frac{d[P]}{dt} = a_2 \frac{[P]^n}{\theta_{a2}^n + [P]^n} + b_2 \frac{\theta_{b2}^m}{\theta_{b2}^m + [G]^m[P]^m} - k_2[P] $$

Extended version in the [report](report/extended_report.md#model-description).

- **[G]** and **[P]** represent normalized gene expression for GATA-1 and PU.1.
- Self-activation and mutual inhibition lead to **bistability**, a key behavior in differentiation.
- Two cases were examined:
  - **Case 1**: Symmetric feedback (a₁ = a₂ = 1)
  - **Case 2**: Asymmetric feedback (a₁ = 5, a₂ = 10)

---

## ⚙️ Methods

We used:

### 🧮 Numerical Methods:
- **LSODE** (R implementation from Schiesser)
- **LSODA** (Python via SciPy)
- **Trapezoidal Method** (custom Python implementation)
- **Radau Solver** (SciPy’s `solve_ivp`)

### 🤖 PINNs:
- Physics-informed loss based on ODE residuals
- Training using PyTorch 
- Case-specific architectures and adaptive weights
- Curriculum learning used to address stiffness in Case 2

---

## 📈 Results

- ✅ **Case 1**: PINNs, LSODA, Radau, and Trapezoidal all showed excellent agreement.
- ⚠️ **Case 2**: PINNs needed deeper architectures and more training epochs to converge, while Radau/LSODA maintained strong performance with minimal tuning.
- 📉 PINNs are flexible and generalize well but are computationally expensive (~1000x slower than numerical solvers in some cases).
- 📊 See full results in the [results](results/) folder.

---

## 💡 Suggestions & Future Work

- Improve training stability with **adaptive sampling** and **gradient-aware point selection**
- Use **transfer learning** across related gene networks
- Test **Fourier-feature PINNs** or **transformer-based solvers**
- Extend the current ODE model to include **spatial terms (PDEs)**

---


