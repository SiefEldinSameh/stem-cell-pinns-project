

# 🧬 Stem Cell Differentiation: Numerical and Machine Learning Approaches to Solving Biological ODEs

## 📌 Overview

This repository explores and compares two powerful approaches to modeling gene regulatory networks that control **stem cell differentiation**:

* ✅ **Numerical Methods**: Trapezoidal Rule, Radau Method, LSODA (via `deSolve`)
* 🤖 **Physics-Informed Neural Networks (PINNs)**: Implemented in **PyTorch**

We analyze the dynamic interaction between transcription factors **PU.1** and **GATA-1** using nonlinear ODEs and evaluate each method’s performance in accuracy, efficiency, and biological insight.

---

## 📄 Full Extended Report

📘 **Looking for all the mathematical derivations, biological background, and in-depth analysis?**
👉 **[Read the Full Report Here → `Untitled-1.md`](./Untitled-1.md)**

This markdown document includes:

* 🔬 Detailed biological context of the PU.1–GATA-1 system
* 🧮 Full ODE formulation with interpretation of each term
* 🧪 Derivation and explanation of all numerical methods used
* 🤖 Full PINN design, training regime, and performance metrics
* 📊 Head-to-head comparison between classical solvers and neural networks
* 🧭 Advanced topics: multi-scale modeling, hybrid solvers, clinical relevance

---

## 🧠 Project Scope

* **Biological Focus**: Hematopoietic stem cells committing to red (erythroid) or white (myeloid) blood cell fates
* **ODE Model**: Captures mutual inhibition and self-activation dynamics of key transcription factors
* **Solver Comparison**: Benchmarking traditional stiff ODE solvers vs. neural solvers (PINNs)

---

## 📂 Repository Structure

```text
📁 stemcell-differentiation/
├── notebooks/
│   ├── numerical_methods.ipynb        # Trapezoidal and Radau implementations
│   ├── pinn_case1_training.ipynb      # PINN model for symmetric case (a1 = a2 = 1)
│   ├── pinn_case2_training.ipynb      # PINN model for asymmetric case (a1 = 5, a2 = 10)
│
├── src/
│   ├── pinn_model.py                  # PINN architecture and loss functions
│   ├── ode_systems.py                 # Biological ODE definitions
│   ├── radau_solver.py                # Custom Radau IIA implementation
│   └── trapezoidal_solver.py          # Fixed-point iterative trapezoidal solver
│
├── results/
│   ├── figures/                       # GATA-1 / PU.1 time-course plots
│   └── metrics/                       # Quantitative benchmarks
│
├── README.md                          # 📘 This file
└── Untitled-1.md                      # 📚 Full detailed write-up
```

---

## 🚀 How to Run

1. **Set up dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run numerical simulations**
   Launch `notebooks/numerical_methods.ipynb` to execute Radau and Trapezoidal solvers.

3. **Train the PINNs**
   Use `pinn_case1_training.ipynb` and `pinn_case2_training.ipynb` for training symmetric and asymmetric scenarios.

4. **Visualize & Compare**
   All output plots and metrics are saved in the `results/` folder for easy comparison.

---

## 🧪 Method Comparison Summary

| Aspect                | Numerical Solvers  | PINNs (PyTorch)            |
| --------------------- | ------------------ | -------------------------- |
| **Accuracy**          | High, controllable | Good, training-dependent   |
| **Speed**             | Very fast          | Slow training, fast eval   |
| **Stiffness Support** | Excellent (Radau)  | Challenging                |
| **Data Integration**  | Difficult          | Natural fit                |
| **Extensibility**     | Limited to ODEs    | Flexible for hybrid models |

---

## 📊 Key Results

| Metric                | Case 1 (Symmetric) | Case 2 (Asymmetric) |
| --------------------- | ------------------ | ------------------- |
| MSE (Numerical)       | \~1e-5             | \~1e-4              |
| MSE (PINNs)           | \~1e-5             | \~1e-4              |
| Training Time (PINNs) | \~3.3 min          | \~6.2 min           |
| Eval Time (PINNs)     | Instant            | Instant             |

---



## 🙏 Acknowledgements

* Inspired by the biological model of PU.1-GATA-1 toggle switches in stem cell differentiation
* Based on foundational systems biology work by Duff et al. (2012)
* Supported by the scientific Python and PyTorch communities

---
