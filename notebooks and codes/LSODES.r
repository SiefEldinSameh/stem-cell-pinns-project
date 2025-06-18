
library("deSolve")  # ODE solver

# -------- ODE Function --------
stem_1 <- function(t, y, parms) {
  G <- y[1]
  P <- y[2]
  
  dGdt <- a1 * G^n / (tha1^n + G^n) + b1 * thb1^m / (thb1^m + G^m * P^m) - k1 * G
  dPdt <- a2 * P^n / (tha2^n + P^n) + b2 * thb2^m / (thb2^m + G^m * P^m) - k2 * P
  
  # Track number of calls
  ncall <<- ncall + 1
  
  return(list(c(dGdt, dPdt)))
}

# -------- Main Program --------
for (ncase in 1:2) {
  
  # Model parameters
  b1 <- 1; b2 <- 1
  tha1 <- 0.5; tha2 <- 0.5
  thb1 <- 0.07; thb2 <- 0.07
  k1 <- 1; k2 <- 1
  n <- 4; m <- 1
  
  # Initial conditions based on case
  if (ncase == 1) {
    G0 <- 1; P0 <- 1; a1 <- 1; a2 <- 1
  } else if (ncase == 2) {
    G0 <- 1; P0 <- 1; a1 <- 5; a2 <- 10
  }
  
  # Time settings
  tf <- 5
  nout <- 26
  tm <- seq(from = 0, to = tf, by = tf / (nout - 1))
  ncall <- 0
  
  # Initialize arrays
  G <- P <- dG <- dP <- numeric(nout)
  G[1] <- G0; P[1] <- P0
  
  # Initial derivatives
  dydt <- stem_1(tm[1], c(G[1], P[1]), NULL)[[1]]
  dG[1] <- dydt[1]; dP[1] <- dydt[2]
  
  # Header
  cat(sprintf("\n ncase = %2d    n = %5.2f   m = %5.2f\n\n", ncase, n, m))
  cat(sprintf("   t       G(t)     P(t)   dG/dt   dP/dt\n"))
  cat(sprintf("%5.2f %10.3f %10.3f %8.3f %8.3f\n", tm[1], G[1], P[1], dG[1], dP[1]))
  
  # ODE solver
  out <- lsodes(times = tm, y = c(G[1], P[1]), func = stem_1, parms = NULL,
                rtol = 1e-8, atol = 1e-8)
  
  for (i in 2:nout) {
    G[i] <- out[i, 2]
    P[i] <- out[i, 3]
    dydt <- stem_1(tm[i], c(G[i], P[i]), NULL)[[1]]
    dG[i] <- dydt[1]
    dP[i] <- dydt[2]
    cat(sprintf("%5.2f %10.3f %10.3f %8.3f %8.3f\n", tm[i], G[i], P[i], dG[i], dP[i]))
  }
  
  cat(sprintf("\n Number of calls to stem_1 = %5d\n\n", ncall))
  
  # Plots
  par(mfrow = c(2, 2))
  plot(tm, G, type = "l", lwd = 2, col = "blue", main = "G(t), LSODES", xlab = "t", ylab = "G(t)")
  plot(tm, P, type = "l", lwd = 2, col = "darkgreen", main = "P(t), LSODES", xlab = "t", ylab = "P(t)")
  plot(tm, dG, type = "l", lwd = 2, col = "red", main = "dG(t)/dt", xlab = "t", ylab = "dG/dt")
  plot(tm, dP, type = "l", lwd = 2, col = "purple", main = "dP(t)/dt", xlab = "t", ylab = "dP/dt")
  
  # -------- Term Analysis (Optional Plotting) --------
  Gterm1 <- Gterm2 <- Gterm3 <- Pterm1 <- Pterm2 <- Pterm3 <- numeric(nout)
  
  for (i in 1:nout) {
    Gterm1[i] <- a1 * G[i]^n / (tha1^n + G[i]^n)
    Gterm2[i] <- b1 * thb1^m / (thb1^m + G[i]^m * P[i]^m)
    Gterm3[i] <- -k1 * G[i]
    Pterm1[i] <- a2 * P[i]^n / (tha2^n + P[i]^n)
    Pterm2[i] <- b2 * thb2^m / (thb2^m + G[i]^m * P[i]^m)
    Pterm3[i] <- -k2 * P[i]
  }
  
  # Plot G terms
  par(mfrow = c(1, 1))
  plot(tm, Gterm1, type = "b", lwd = 2, pch = "1", ylim = c(-5, 5),
       main = "Gterm1, Gterm2, Gterm3, dG/dt vs t", xlab = "t", ylab = "Terms")
  lines(tm, Gterm2, type = "b", lwd = 2, pch = "2")
  lines(tm, Gterm3, type = "b", lwd = 2, pch = "3")
  lines(tm, dG,      type = "b", lwd = 2, pch = "4")
  
  # Plot P terms
  plot(tm, Pterm1, type = "b", lwd = 2, pch = "1", ylim = c(-10, 10),
       main = "Pterm1, Pterm2, Pterm3, dP/dt vs t", xlab = "t", ylab = "Terms")
  lines(tm, Pterm2, type = "b", lwd = 2, pch = "2")
  lines(tm, Pterm3, type = "b", lwd = 2, pch = "3")
  lines(tm, dP,      type = "b", lwd = 2, pch = "4")
}
