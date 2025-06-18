import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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
    
    return [dGdt, dPdt]

def main():
    """
    Main program for stem cell differentiation model
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
            a1 = 4
            a2 = 36
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
        
        # Solve ODE system using solve_ivp with LSODA method (equivalent to lsodes)
        sol = solve_ivp(lambda t, y: stem_1(t, y, params), 
                       t_span, y0, t_eval=t_eval, 
                       method='LSODA', 
                       rtol=1e-8, atol=1e-8)
        
        # End timing for ODE solving
        solve_end_time = time.time()
        ode_solve_time = solve_end_time - solve_start_time
        
        # Extract solutions
        G = sol.y[0]
        P = sol.y[1]
        tm = sol.t
        
        # Calculate derivatives at each time point
        dG = np.zeros(nout)
        dP = np.zeros(nout)
        
        for i in range(nout):
            dydt = stem_1(tm[i], [G[i], P[i]], params)
            dG[i] = dydt[0]
            dP[i] = dydt[1]
        
        # Display numerical output (skip initial point since already displayed)
        for i in range(1, nout):
            print(f"{tm[i]:5.2f}{G[i]:10.3f}{P[i]:10.3f}{dG[i]:10.3f}{dP[i]:10.3f}")
        
        print(f"\n ncall = {ncall:5d}")
        print(f"ODE solving time for case {ncase}: {ode_solve_time:.6f} seconds")
        
        # Create plots - Four plots for G(t), P(t), dG(t)/dt, dP(t)/dt vs t
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Stem Cell Differentiation Model - Case {ncase}', fontsize=14)
        
        # G(t)
        ax1.plot(tm, G, 'b-', linewidth=2)
        ax1.set_xlabel('t')
        ax1.set_ylabel('G(t)')
        ax1.set_xlim(0, 5)
        ax1.set_title('G(t), LSODA')
        ax1.grid(True, alpha=0.3)
        
        # P(t)
        ax2.plot(tm, P, 'r-', linewidth=2)
        ax2.set_xlabel('t')
        ax2.set_ylabel('P(t)')
        ax2.set_xlim(0, 5)
        ax2.set_title('P(t), LSODA')
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
            analyze_ode_terms(tm, G, P, params, nout)
    
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
    plt.title('Gterm1,Gterm2,Gterm3,dG/dt vs t')
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
    plt.title('Pterm1,Pterm2,Pterm3,dP/dt vs t')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking show

if __name__ == "__main__":
    main()