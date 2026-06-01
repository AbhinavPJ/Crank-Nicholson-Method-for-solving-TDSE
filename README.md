# Crank Nicolson Method for solving the Time Dependent Schrödinger's Equation in Python + Pretty Poster

<img width="1053" height="747" alt="image" src="https://github.com/user-attachments/assets/0be733c3-377b-4ebe-9ad9-075e8cf7f76f" />

This repository provides a numerical simulation of the time evolution of a quantum mechanical wavepacket under various physical potentials. The implementation demonstrates time-reversal symmetry of the Schrödinger equation and utilizes highly efficient linear algebra techniques.

## Visualization

[https://github.com/user-attachments/assets/343d255b-dbd6-49f5-ab4e-6c8a29874aaa](https://github.com/user-attachments/assets/343d255b-dbd6-49f5-ab4e-6c8a29874aaa)

## Methodology

The simulation solves the 1D Time-Dependent Schrödinger Equation (TDSE) using the Crank-Nicolson method.

* **Discretization:** Space and time are discretized, and the spatial second derivative is approximated using a standard centered finite difference.


* **Stability and Unitarity:** By averaging the Hamiltonian at the temporal midpoint (equivalent to a Padé [1,1] approximant), the Crank-Nicolson scheme is unconditionally stable. It is exactly unitary, meaning it strictly conserves the physical probability density over time.


* **Efficiency:** The discretization produces tridiagonal matrices. Instead of using dense LU decomposition which costs $O(N^3)$, the system is solved using the Thomas algorithm. This specialized Gaussian elimination reduces the computational cost to $O(N)$ time per step.



## Supported Potentials

The numerical solver analyzes four paradigmatic quantum scenarios:

* **Free Particle:** The wavepacket spreads due to dispersion.


* **Harmonic Oscillator:** The wavepacket oscillates around equilibrium without spreading.


* **Particle in a Box:** Demonstrates reflection and interference against hard walls.


* **Potential Barrier:** Demonstrates quantum tunneling, where the particle transmits through the barrier.



## Resources

* [View Poster PDF](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fuser-attachments%2Ffiles%2F28478471%2Fmain.pdf) - A complete mathematical derivation covering operator formulations, linear systems, and scheme stability.
