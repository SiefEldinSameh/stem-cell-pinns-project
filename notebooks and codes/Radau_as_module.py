import numpy as np
from scipy.linalg import solve
import time

# Global variable to count function calls
ncall = 0

def stem_1(t, y, params):
    """
    ODE system for stem cell differentiation model
    Equations (5.1) from the document
    """
    global ncall
    
    # Extract parameters
    a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m = params
    
    # Transfer dependent variable vector to problem variables
    G = y[0]
    P = y[1]
    
    # Stem cell model ODEs (equations 5.1)
    dGdt = a1 * G**n / (tha1**n + G**n) + b1 * thb1**m / (thb1**m + G**m * P**m) - k1 * G
    dPdt = a2 * P**n / (tha2**n + P**n) + b2 * thb2**m / (thb2**m + G**m * P**m) - k2 * P
    
    # Increment calls to stem_1
    ncall += 1
    
    return np.array([dGdt, dPdt])

def jacobian(t, y, params):
    """
    Compute the Jacobian matrix of the ODE system
    """
    a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m = params
    G, P = y[0], y[1]
    
    # Avoid division by zero
    eps = 1e-15
    G = max(G, eps)
    P = max(P, eps)
    
    # Common terms
    tha1n_Gn = tha1**n + G**n
    tha2n_Pn = tha2**n + P**n
    thb1m_GmPm = thb1**m + G**m * P**m
    thb2m_GmPm = thb2**m + G**m * P**m
    
    # Partial derivatives for dG/dt
    dG_dG = (a1 * n * G**(n-1) * tha1**n) / tha1n_Gn**2 - \
            (b1 * thb1**m * m * G**(m-1) * P**m) / thb1m_GmPm**2 - k1
    
    dG_dP = -(b1 * thb1**m * m * G**m * P**(m-1)) / thb1m_GmPm**2
    
    # Partial derivatives for dP/dt
    dP_dG = -(b2 * thb2**m * m * G**(m-1) * P**m) / thb2m_GmPm**2
    
    dP_dP = (a2 * n * P**(n-1) * tha2**n) / tha2n_Pn**2 - \
            (b2 * thb2**m * m * G**m * P**(m-1)) / thb2m_GmPm**2 - k2
    
    return np.array([[dG_dG, dG_dP], [dP_dG, dP_dP]])

class RadauIntegrator:
    """
    Custom implementation of Radau IIA method (3-stage, 5th order)
    """
    
    def __init__(self, rtol=1e-8, atol=1e-8, max_iter=50):
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        
        # Radau IIA coefficients (3-stage, 5th order)
        # Butcher tableau coefficients
        self.c = np.array([
            (4 - np.sqrt(6)) / 10,
            (4 + np.sqrt(6)) / 10,
            1.0
        ])
        
        self.A = np.array([
            [(88 - 7*np.sqrt(6)) / 360, (296 - 169*np.sqrt(6)) / 1800, (-2 + 3*np.sqrt(6)) / 225],
            [(296 + 169*np.sqrt(6)) / 1800, (88 + 7*np.sqrt(6)) / 360, (-2 - 3*np.sqrt(6)) / 225],
            [(16 - np.sqrt(6)) / 36, (16 + np.sqrt(6)) / 36, 1/9]
        ])
        
        self.b = np.array([
            (16 - np.sqrt(6)) / 36,
            (16 + np.sqrt(6)) / 36,
            1/9
        ])
        
        # Error estimation coefficients (4th order embedded method)
        self.b_hat = np.array([
            (16 - np.sqrt(6)) / 36,
            (16 + np.sqrt(6)) / 36,
            1/9
        ])
        
        self.s = 3  # number of stages
        
    def step(self, f, jac, t, y, h, params):
        """
        Perform one step of Radau IIA method
        """
        n = len(y)
        
        # Initialize stage values
        Y = np.zeros((self.s, n))
        for i in range(self.s):
            Y[i] = y.copy()
        
        # Newton iteration for implicit stages
        for iter_count in range(self.max_iter):
            # Compute function values at stage points
            F = np.zeros((self.s, n))
            for i in range(self.s):
                F[i] = f(t + self.c[i] * h, Y[i], params)
            
            # Build the nonlinear system residual
            R = np.zeros((self.s, n))
            for i in range(self.s):
                R[i] = Y[i] - y
                for j in range(self.s):
                    R[i] -= h * self.A[i, j] * F[j]
            
            # Check convergence
            max_residual = np.max(np.abs(R))
            if max_residual < self.rtol * max(1.0, np.max(np.abs(y))) + self.atol:
                break
            
            # Compute Jacobians for Newton system
            J = np.zeros((self.s * n, self.s * n))
            for i in range(self.s):
                J_f = jac(t + self.c[i] * h, Y[i], params)
                for k in range(n):
                    for l in range(n):
                        # Diagonal blocks: I - h * A[i,i] * J_f
                        J[i*n + k, i*n + l] = (1.0 if k == l else 0.0) - h * self.A[i, i] * J_f[k, l]
                        
                        # Off-diagonal blocks: -h * A[i,j] * J_f
                        for j in range(self.s):
                            if i != j:
                                J[i*n + k, j*n + l] = -h * self.A[i, j] * J_f[k, l]
            
            # Solve Newton system
            R_flat = R.flatten()
            try:
                delta = solve(J, -R_flat)
                delta = delta.reshape((self.s, n))
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                delta = np.linalg.pinv(J) @ (-R_flat)
                delta = delta.reshape((self.s, n))
            
            # Update stage values
            Y += delta
        
        # Compute new solution
        y_new = y.copy()
        for i in range(self.s):
            F[i] = f(t + self.c[i] * h, Y[i], params)
            y_new += h * self.b[i] * F[i]
        
        # Error estimation (simplified)
        error = np.zeros(n)
        for i in range(self.s):
            error += h * (self.b[i] - self.b_hat[i]) * F[i]
        error_norm = np.linalg.norm(error) / max(1.0, np.linalg.norm(y))
        
        return y_new, error_norm
    
    def adaptive_step_size(self, error_norm, h, safety_factor=0.9, min_factor=0.2, max_factor=5.0):
        """
        Adaptive step size control
        """
        if error_norm == 0:
            return h * max_factor
        
        # PI controller for step size
        factor = safety_factor * (self.rtol / error_norm) ** (1/5)
        factor = max(min_factor, min(max_factor, factor))
        
        return h * factor
    
    def integrate(self, f, jac, t_span, y0, t_eval, params):
        """
        Integrate the ODE system using custom Radau IIA method
        """
        t_start, t_end = t_span
        y = np.array(y0, dtype=float)
        t = t_start
        
        # Use specified evaluation points
        t_out = []
        y_out = []
        
        eval_idx = 0
        h = (t_end - t_start) / 1000  # Initial step size
        
        # Add initial point if it's in t_eval
        if len(t_eval) > 0 and abs(t_eval[0] - t) < 1e-12:
            t_out.append(t)
            y_out.append(y.copy())
            eval_idx = 1
        
        while eval_idx < len(t_eval) and t < t_end:
            t_target = t_eval[eval_idx]
            
            while t < t_target:
                h_try = min(h, t_target - t)
                
                # Take a step
                y_new, error_norm = self.step(f, jac, t, y, h_try, params)
                
                # Check if step is acceptable
                if error_norm <= 1.0 or h_try <= 1e-12:
                    # Accept step
                    t += h_try
                    y = y_new
                else:
                    # Reject step and reduce step size
                    h = self.adaptive_step_size(error_norm, h_try)
                    continue
                
                # Adapt step size for next step
                if h_try == h:  # Only adapt if we used the full step
                    h = self.adaptive_step_size(error_norm, h)
            
            # Interpolate to exact evaluation point if needed
            if abs(t - t_target) > 1e-12:
                # Simple linear interpolation (could be improved)
                alpha = (t_target - (t - h_try)) / h_try if h_try > 0 else 0
                y_interp = y + alpha * (y - y_out[-1] if y_out else np.zeros_like(y))
                t_out.append(t_target)
                y_out.append(y_interp)
            else:
                t_out.append(t)
                y_out.append(y.copy())
            
            eval_idx += 1
        
        return np.array(t_out), np.array(y_out).T

def solve_stem_cell_radau(case_num, t_eval):
    """
    Solve stem cell differentiation model using custom Radau method
    
    Parameters:
    -----------
    case_num : int
        Case number (1 or 2) to select parameter set
    t_eval : array_like
        Time points where solution is evaluated
    
    Returns:
    --------
    G_radau : ndarray
        Array of G values at t_eval points
    P_radau : ndarray  
        Array of P values at t_eval points
    elapsed_time : float
        Computation time in seconds
    """
    global ncall
    
    # Reset call counter
    ncall = 0
    
    # Model parameters (common to both cases)
    b1 = 1
    b2 = 1
    tha1 = 0.5
    tha2 = 0.5
    thb1 = 0.07
    thb2 = 0.07
    k1 = 1
    k2 = 1
    n = 4
    m = 1
    
    # Case-specific parameters and initial conditions
    if case_num == 1:
        G0 = 1
        P0 = 1
        a1 = 1
        a2 = 1
    elif case_num == 2:
        G0 = 1
        P0 = 1
        a1 = 5
        a2 = 10
    else:
        raise ValueError("case_num must be 1 or 2")
    
    # Setup
    t_span = (t_eval[0], t_eval[-1])
    y0 = [G0, P0]
    params = [a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m]
    
    # Start timing
    start_time = time.time()
    
    # Create custom Radau integrator
    radau = RadauIntegrator(rtol=1e-8, atol=1e-8)
    
    # Solve ODE system using custom Radau method
    tm, y_sol = radau.integrate(stem_1, jacobian, t_span, y0, t_eval, params)
    
    # End timing
    elapsed_time = time.time() - start_time
    
    # Extract solutions
    G_radau = y_sol[0]
    P_radau = y_sol[1]
    
    return G_radau, P_radau, elapsed_time

def get_function_call_count():
    """
    Get the number of function calls made during integration
    
    Returns:
    --------
    int : Number of calls to stem_1 function
    """
    global ncall
    return ncall

def reset_function_call_count():
    """
    Reset the function call counter
    """
    global ncall
    ncall = 0

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Example: solve case 1 over time interval [0, 5] with 26 points
    t_eval = np.linspace(0, 5, 3)
    
    # Solve case 1
    G1, P1, time1 = solve_stem_cell_radau(1, t_eval)
    print(f"Case 1: Solved in {time1:.6f} seconds, {get_function_call_count()} function calls")
    
    # Solve case 2  
    G2, P2, time2 = solve_stem_cell_radau(2, t_eval)
    print(f"Case 2: Solved in {time2:.6f} seconds, {get_function_call_count()} function calls")
    
    # Print first few values for verification
    print(f"\nCase 1 - First few G values: {G1[:5]}")
    print(f"Case 1 - First few P values: {P1[:5]}")
    print(f"\nCase 2 - First few G values: {G2[:5]}")
    print(f"Case 2 - First few P values: {P2[:5]}")