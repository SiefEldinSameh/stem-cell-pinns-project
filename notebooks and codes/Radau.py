import numpy as np
import matplotlib.pyplot as plt
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
        
        # Initialize output arrays
        if t_eval is None:
            # Use adaptive time stepping
            t_out = [t]
            y_out = [y.copy()]
            h = (t_end - t_start) / 100  # Initial step size
            
            while t < t_end:
                h = min(h, t_end - t)
                
                # Take a step
                y_new, error_norm = self.step(f, jac, t, y, h, params)
                
                # Check if step is acceptable
                if error_norm <= 1.0 or h <= 1e-12:
                    # Accept step
                    t += h
                    y = y_new
                    t_out.append(t)
                    y_out.append(y.copy())
                
                # Adapt step size
                h = self.adaptive_step_size(error_norm, h)
                
            return np.array(t_out), np.array(y_out).T
        
        else:
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

def main():
    """
    Main program for stem cell differentiation model using custom Radau method
    """
    global ncall
    
    # Step through cases
    for ncase in range(1, 3):  # ncase = 1, 2
        print(f"\n=== Case {ncase} ===")
        
        # Reset call counter
        ncall = 0
        
        # Model parameters
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
        if ncase == 1:
            G0 = 1
            P0 = 1
            a1 = 1
            a2 = 1
        elif ncase == 2:
            G0 = 1
            P0 = 1
            a1 = 5
            a2 = 10
        
        # Write selected parameters and heading
        print(f"\n ncase = {ncase:2d} n = {n:5.2f} m = {m:5.2f}\n")
        print("     t         G         P      dG/dt      dP/dt")
        
        # Initial condition and time setup
        tf = 5
        nout = 26
        t_span = (0, tf)
        t_eval = np.linspace(0, tf, nout)
        y0 = [G0, P0]
        
        # Parameters for ODE function
        params = [a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m]
        
        # Initial derivatives
        dydt_initial = stem_1(0, y0, params)
        dG_initial = dydt_initial[0]
        dP_initial = dydt_initial[1]
        
        # Display initial variables
        print(f"{t_eval[0]:5.2f}{y0[0]:10.3f}{y0[1]:10.3f}{dG_initial:10.3f}{dP_initial:10.3f}")
        
        # Start timing for ODE solving only
        solve_start_time = time.time()
        
        # Create custom Radau integrator
        radau = RadauIntegrator(rtol=1e-8, atol=1e-8)
        
        # Solve ODE system using custom Radau method
        tm, y_sol = radau.integrate(stem_1, jacobian, t_span, y0, t_eval, params)
        
        # End timing for ODE solving
        solve_end_time = time.time()
        ode_solve_time = solve_end_time - solve_start_time
        
        # Extract solutions
        G = y_sol[0]
        P = y_sol[1]
        
        # Calculate derivatives at each time point
        dG = np.zeros(len(tm))
        dP = np.zeros(len(tm))
        
        for i in range(len(tm)):
            dydt = stem_1(tm[i], [G[i], P[i]], params)
            dG[i] = dydt[0]
            dP[i] = dydt[1]
        
        # Display numerical output (skip initial point since already displayed)
        for i in range(1, len(tm)):
            print(f"{tm[i]:5.2f}{G[i]:10.6f}{P[i]:10.6f}{dG[i]:10.6f}{dP[i]:10.6f}")
        
        print(f"\n ncall = {ncall:5d}")
        print(f"ODE solving time for case {ncase}: {ode_solve_time:.6f} seconds")
        
        # Create plots - Four plots for G(t), P(t), dG(t)/dt, dP(t)/dt vs t
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Stem Cell Differentiation Model - Case {ncase} (Custom Radau)', fontsize=14)
        
        # G(t)
        ax1.plot(tm, G, 'b-', linewidth=2)
        ax1.set_xlabel('t')
        ax1.set_ylabel('G(t)')
        ax1.set_xlim(0, 5)
        ax1.set_title('G(t), Custom Radau')
        ax1.grid(True, alpha=0.3)
        
        # P(t)
        ax2.plot(tm, P, 'r-', linewidth=2)
        ax2.set_xlabel('t')
        ax2.set_ylabel('P(t)')
        ax2.set_xlim(0, 5)
        ax2.set_title('P(t), Custom Radau')
        ax2.grid(True, alpha=0.3)
        
        # dG(t)/dt
        ax3.plot(tm, dG, 'g-', linewidth=2)
        ax3.set_xlabel('t')
        ax3.set_ylabel('dG(t)/dt')
        ax3.set_xlim(0, 5)
        ax3.set_title('dG(t)/dt')
        ax3.grid(True, alpha=0.3)
        
        # dP(t)/dt
        ax4.plot(tm, dP, 'm-', linewidth=2)
        ax4.set_xlabel('t')
        ax4.set_ylabel('dP(t)/dt')
        ax4.set_xlim(0, 5)
        ax4.set_title('dP(t)/dt')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking show
        
        # Analysis of the terms in the ODEs (for case 2)
        if ncase == 2:
            analyze_ode_terms(tm, G, P, params, len(tm))
    
    # Close all plots and terminate
    plt.show()  # Show all plots

def analyze_ode_terms(tm, G, P, params, nout):
    """
    Analyze individual terms in the ODEs
    """
    a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2, n, m = params
    
    # Declare arrays for ODE analysis
    Gterm1 = np.zeros(nout)
    Gterm2 = np.zeros(nout)
    Gterm3 = np.zeros(nout)
    Pterm1 = np.zeros(nout)
    Pterm2 = np.zeros(nout)
    Pterm3 = np.zeros(nout)
    dG = np.zeros(nout)
    dP = np.zeros(nout)
    
    # Compute and save the RHS terms of the ODEs
    for i in range(nout):
        Gterm1[i] = a1 * G[i]**n / (tha1**n + G[i]**n)
        Gterm2[i] = b1 * thb1**m / (thb1**m + G[i]**m * P[i]**m)
        Gterm3[i] = -k1 * G[i]
        
        Pterm1[i] = a2 * P[i]**n / (tha2**n + P[i]**n)
        Pterm2[i] = b2 * thb2**m / (thb2**m + G[i]**m * P[i]**m)
        Pterm3[i] = -k2 * P[i]
        
        # Calculate derivatives for comparison
        dG[i] = Gterm1[i] + Gterm2[i] + Gterm3[i]
        dP[i] = Pterm1[i] + Pterm2[i] + Pterm3[i]
    
    # Plot the terms of the G(t) ODE
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(tm, Gterm1, 'o-', label='Gterm1', markersize=4, linewidth=2)
    plt.plot(tm, Gterm2, 's-', label='Gterm2', markersize=4, linewidth=2)
    plt.plot(tm, Gterm3, '^-', label='Gterm3', markersize=4, linewidth=2)
    plt.plot(tm, dG, 'v-', label='dG/dt', markersize=4, linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Gterm1,Gterm2,Gterm3,dG/dt')
    plt.xlim(0, 5)
    plt.ylim(-5, 5)
    plt.title('Gterm1,Gterm2,Gterm3,dG/dt vs t (Custom Radau)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot the terms of the P(t) ODE
    plt.subplot(1, 2, 2)
    plt.plot(tm, Pterm1, 'o-', label='Pterm1', markersize=4, linewidth=2)
    plt.plot(tm, Pterm2, 's-', label='Pterm2', markersize=4, linewidth=2)
    plt.plot(tm, Pterm3, '^-', label='Pterm3', markersize=4, linewidth=2)
    plt.plot(tm, dP, 'v-', label='dP/dt', markersize=4, linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Pterm1,Pterm2,Pterm3,dP/dt')
    plt.xlim(0, 5)
    plt.ylim(-10, 10)
    plt.title('Pterm1,Pterm2,Pterm3,dP/dt vs t (Custom Radau)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking show

if __name__ == "__main__":
    main()