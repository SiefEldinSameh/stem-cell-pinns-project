

# 🧬 Stem Cell Differentiation: Numerical and Machine Learning Approaches to Solving Biological ODEs

## 📌 Overview

This repository explores and compares two powerful approaches to modeling gene regulatory networks that control **stem cell differentiation**:

* ✅ **Numerical Methods**: Trapezoidal Rule, Radau Method, LSODA (via `deSolve`)
* 🤖 **Physics-Informed Neural Networks (PINNs)**: Implemented in **PyTorch**

We analyze the dynamic interaction between transcription factors **PU.1** and **GATA-1** using nonlinear ODEs and evaluate each method’s performance in accuracy, efficiency, and biological insight.

---

## 📄 Full Extended Report

📘 **Looking for all the mathematical derivations, biological background, and in-depth analysis?**
👉 **[Read the Full Report Here → `Full Extended Version.md`](./report/Full_Extended_version.md)**

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
📁 stem-cell-pinns-project/
├── notebooks and codes/
│   ├── LSODA.py                       # LSODA numerical solver (Python)
│   ├── LSODES.r                       # LSODES solver implementation (R)
│   ├── PINNS.ipynb                    # PINN training 
│   ├── PINNS VS Numerical.ipynb      # Comparison between PINNs and numerical solvers
│   ├── Radau.py                       # Radau method implementation
│   ├── Radau_as_module.py            # Modular Radau solver
│   ├── Trapzoidal.py                 # Trapezoidal solver implementation
│   └── Trapzoidal_as_module.py       # Modular Trapezoidal method
│
├── report/
│   └── Full_Extended_version.md      # 📚 Complete project write-up and analysis
│
├── results/
│   ├── both/                         # Comparative plots for both cases
│   ├── case1/                        # Figures for Case 1 (a1 = a2 = 1)
│   └── case2/                        # Figures for Case 2 (a1 = 5, a2 = 10)
│
├── README.md                         # 📘 Project overview and instructions
└── LICENSE                           # 📄 Licensing information
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
