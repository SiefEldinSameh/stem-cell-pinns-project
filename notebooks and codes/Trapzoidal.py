
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import time

def stem_1(t, y, params):
    a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m = params
    G, P = y
    dGdt = a1 * G**n / (tha1**n + G**n) + b1 * thb1**m / (thb1**m + G**m * P**m) - k1 * G
    dPdt = a2 * P**n / (tha2**n + P**n) + b2 * thb2**m / (thb2**m + G**m * P**m) - k2 * P
    return np.array([dGdt, dPdt])

def trapezoidal_solver_2d(ode_func: Callable, t_span, y0, params, n_steps, tol=1e-6, max_iter=100):
    total_iter = 0
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t_vals = np.linspace(t0, tf, n_steps + 1)
    y_vals = np.zeros((len(y0), n_steps + 1))
    y_vals[:, 0] = y0

    for i in range(n_steps):
        t_n = t_vals[i]
        y_n = y_vals[:, i]
        f_n = ode_func(t_n, y_n, params)
        y_guess = y_n + h * f_n
        for _ in range(max_iter):
            total_iter += 1
            f_guess = ode_func(t_n + h, y_guess, params)
            y_next = y_n + (h / 2) * (f_n + f_guess)
            if np.linalg.norm(y_next - y_guess) < tol:
                break
            y_guess = y_next
        y_vals[:, i + 1] = y_next
    return t_vals, y_vals, total_iter

def plot_case(t_vals, y_vals, params, case_title):
    G_vals, P_vals = y_vals
    dG_vals = []
    dP_vals = []

    print(f"\n=== {case_title} ===")
    print(f"{'t':>8}{'G':>12}{'P':>12}{'dG/dt':>12}{'dP/dt':>12}")

    for t, G, P in zip(t_vals, G_vals, P_vals):
        dG, dP = stem_1(t, [G, P], params)
        dG_vals.append(dG)
        dP_vals.append(dP)
        print(f"{t:8.3f}{G:12.6f}{P:12.6f}{dG:12.6f}{dP:12.6f}")

    plt.figure(figsize=(12, 8))
    plt.suptitle(case_title)

    plt.subplot(2, 2, 1)
    plt.plot(t_vals, G_vals, 'b')
    plt.title("G(t), Trapezoidal")
    plt.xlabel("t")
    plt.ylabel("G(t)")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(t_vals, P_vals, 'r')
    plt.title("P(t), Trapezoidal")
    plt.xlabel("t")
    plt.ylabel("P(t)")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(t_vals, dG_vals, 'g')
    plt.title("dG(t)/dt")
    plt.xlabel("t")
    plt.ylabel("dG/dt")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(t_vals, dP_vals, 'm')
    plt.title("dP(t)/dt")
    plt.xlabel("t")
    plt.ylabel("dP/dt")
    plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)

def analyze_ode_terms(tm, G, P, params, nout):
    a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m = params
    Gterm1 = np.zeros(nout)
    Gterm2 = np.zeros(nout)
    Gterm3 = np.zeros(nout)
    Pterm1 = np.zeros(nout)
    Pterm2 = np.zeros(nout)
    Pterm3 = np.zeros(nout)
    dG = np.zeros(nout)
    dP = np.zeros(nout)

    for i in range(nout):
        Gterm1[i] = a1 * G[i] ** n / (tha1 ** n + G[i] ** n)
        Gterm2[i] = b1 * thb1 ** m / (thb1 ** m + G[i] ** m * P[i] ** m)
        Gterm3[i] = -k1 * G[i]
        Pterm1[i] = a2 * P[i] ** n / (tha2 ** n + P[i] ** n)
        Pterm2[i] = b2 * thb2 ** m / (thb2 ** m + G[i] ** m * P[i] ** m)
        Pterm3[i] = -k2 * P[i]
        dG[i] = Gterm1[i] + Gterm2[i] + Gterm3[i]
        dP[i] = Pterm1[i] + Pterm2[i] + Pterm3[i]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tm, Gterm1, 'o-', label='Gterm1', markersize=4, linewidth=2)
    plt.plot(tm, Gterm2, 's-', label='Gterm2', markersize=4, linewidth=2)
    plt.plot(tm, Gterm3, '^-', label='Gterm3', markersize=4, linewidth=2)
    plt.plot(tm, dG, 'v-', label='dG/dt', markersize=4, linewidth=2)
    plt.xlabel('t')
    plt.ylabel('G terms')
    plt.title('G terms and dG/dt')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(tm, Pterm1, 'o-', label='Pterm1', markersize=4, linewidth=2)
    plt.plot(tm, Pterm2, 's-', label='Pterm2', markersize=4, linewidth=2)
    plt.plot(tm, Pterm3, '^-', label='Pterm3', markersize=4, linewidth=2)
    plt.plot(tm, dP, 'v-', label='dP/dt', markersize=4, linewidth=2)
    plt.xlabel('t')
    plt.ylabel('P terms')
    plt.title('P terms and dP/dt')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)

if __name__ == "__main__":
    b1 = b2 = 1
    tha1 = tha2 = 0.5
    thb1 = thb2 = 0.07
    k1 = k2 = 1
    n = 4
    m = 1
    G0 = P0 = 1
    y0 = [G0, P0]
    t_span = (0, 5)
    n_steps = 25

    a1, a2 = 1, 1
    params1 = [a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m]
    start = time.time()
    t_vals1, y_vals1, iters1 = trapezoidal_solver_2d(stem_1, t_span, y0, params1, n_steps)
    plot_case(t_vals1, y_vals1, params1, "Stem Cell Model - Case 1 (a1=1, a2=1)")
    print(f"\nCase 1 solved in {time.time() - start:.6f} sec\n ncall = {iters1}")

    a1, a2 = 5, 10
    params2 = [a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m]
    start = time.time()
    t_vals2, y_vals2, iters2 = trapezoidal_solver_2d(stem_1, t_span, y0, params2, n_steps)
    plot_case(t_vals2, y_vals2, params2, "Stem Cell Model - Case 2 (a1=5, a2=10)")
    print(f"\nCase 2 solved in {time.time() - start:.6f} sec \n ncall = {iters2}")

    analyze_ode_terms(t_vals2, y_vals2[0], y_vals2[1], params2, len(t_vals2))
    plt.show()
