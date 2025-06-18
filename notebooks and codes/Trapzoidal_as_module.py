import numpy as np
import time
from typing import Callable, Tuple, Union

def stem_cell_ode(t: float, y: np.ndarray, params: list) -> np.ndarray:
    """
    Stem cell model ODE system.
    
    Parameters:
    t: time (not used in autonomous system)
    y: [G, P] state vector
    params: [a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m]
    
    Returns:
    dy/dt: [dG/dt, dP/dt]
    """
    a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m = params
    G, P = y
    
    # G equation terms
    dGdt = (a1 * G**n / (tha1**n + G**n) + 
            b1 * thb1**m / (thb1**m + G**m * P**m) - 
            k1 * G)
    
    # P equation terms
    dPdt = (a2 * P**n / (tha2**n + P**n) + 
            b2 * thb2**m / (thb2**m + G**m * P**m) - 
            k2 * P)
    
    return np.array([dGdt, dPdt])

def trapezoidal_method(ode_func: Callable, 
                      t_span: Tuple[float, float], 
                      y0: Union[list, np.ndarray], 
                      params: list, 
                      t_eval: np.ndarray = None,
                      n_steps: int = None,
                      tol: float = 1e-6, 
                      max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve ODE system using the trapezoidal method.
    
    Parameters:
    ode_func: Function defining the ODE system
    t_span: (t0, tf) time span
    y0: Initial conditions [G0, P0]
    params: ODE system parameters
    t_eval: Specific time points to evaluate (optional)
    n_steps: Number of steps if t_eval not provided
    tol: Convergence tolerance for implicit solver
    max_iter: Maximum iterations for implicit solver
    
    Returns:
    G_trap: Array of G values at t_eval points
    P_trap: Array of P values at t_eval points
    elapsed_time: Computation time in seconds
    """
    start_time = time.time()
    
    t0, tf = t_span
    y0 = np.array(y0)
    
    # Set up time points
    if t_eval is not None:
        t_vals = np.array(t_eval)
        # Ensure t_eval is sorted and within bounds
        t_vals = t_vals[(t_vals >= t0) & (t_vals <= tf)]
        if len(t_vals) == 0:
            raise ValueError("No valid evaluation points in t_span")
    elif n_steps is not None:
        t_vals = np.linspace(t0, tf, n_steps + 1)
    else:
        raise ValueError("Either t_eval or n_steps must be provided")
    
    n_points = len(t_vals)
    y_vals = np.zeros((len(y0), n_points))
    y_vals[:, 0] = y0
    
    # Solve step by step
    for i in range(n_points - 1):
        t_curr = t_vals[i]
        y_curr = y_vals[:, i]
        t_next = t_vals[i + 1]
        h = t_next - t_curr
        
        # Explicit predictor (forward Euler)
        f_curr = ode_func(t_curr, y_curr, params)
        y_pred = y_curr + h * f_curr
        
        # Implicit corrector (trapezoidal rule)
        y_next = y_pred.copy()
        for iter_count in range(max_iter):
            f_next = ode_func(t_next, y_next, params)
            y_new = y_curr + (h / 2) * (f_curr + f_next)
            
            # Check convergence
            if np.linalg.norm(y_new - y_next) < tol:
                y_next = y_new
                break
            y_next = y_new
        else:
            print(f"Warning: Maximum iterations reached at step {i+1}")
        
        y_vals[:, i + 1] = y_next
    
    elapsed_time = time.time() - start_time
    
    # Extract G and P values
    G_trap = y_vals[0, :]
    P_trap = y_vals[1, :]
    
    return G_trap, P_trap, elapsed_time

def solve_stem_cell_model(a1: float, a2: float, 
                         G0: float = 1.0, P0: float = 1.0,
                         t_span: Tuple[float, float] = (0, 5),
                         t_eval: np.ndarray = None,
                         n_steps: int = 25,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convenience function to solve the stem cell model with default parameters.
    
    Parameters:
    a1, a2: Production rate parameters
    G0, P0: Initial conditions
    t_span: Time span
    t_eval: Evaluation time points
    n_steps: Number of steps (if t_eval not provided)
    **kwargs: Additional parameters for trapezoidal_method
    
    Returns:
    G_trap: Array of G values at t_eval points
    P_trap: Array of P values at t_eval points
    elapsed_time: Computation time in seconds
    """
    # Default parameters
    b1 = b2 = 1.0
    tha1 = tha2 = 0.5
    thb1 = thb2 = 0.07
    k1 = k2 = 1.0
    n = 4
    m = 1
    
    params = [a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m]
    y0 = [G0, P0]
    
    return trapezoidal_method(stem_cell_ode, t_span, y0, params, 
                             t_eval=t_eval, n_steps=n_steps, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Test Case 1: a1=1, a2=1
    print("=== Test Case 1: a1=1, a2=1 ===")
    G1, P1, time1 = solve_stem_cell_model(a1=1, a2=1, t_eval=)
    print(f"Solved in {time1:.6f} seconds")
    print(f"Final values: G={G1[-1]:.6f}, P={P1[-1]:.6f}")
    print(len(G1))
    
    # Test Case 2: a1=5, a2=10
    print("\n=== Test Case 2: a1=5, a2=10 ===")
    G2, P2, time2 = solve_stem_cell_model(a1=5, a2=10, n_steps=25)
    print(f"Solved in {time2:.6f} seconds")
    print(f"Final values: G={G2[-1]:.6f}, P={P2[-1]:.6f}")
    
    # Test with custom t_eval
    print("\n=== Test with custom evaluation points ===")
    t_custom = np.array([0, 1, 2, 3, 4, 5])
    G3, P3, time3 = solve_stem_cell_model(a1=1, a2=1, t_eval=t_custom)
    print(f"Solved in {time3:.6f} seconds")
    print("Results at custom points:")
    for i, t in enumerate(t_custom):
        print(f"t={t}: G={G3[i]:.6f}, P={P3[i]:.6f}")