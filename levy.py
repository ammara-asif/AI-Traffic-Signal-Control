import numpy as np
from math import gamma, sin, pi

"""
Generate Levy flight random numbers
(Eq 5 and 6)

Parameters:
pop (int): Number of rows in output
m (int): Number of columns in output
omega (float): Levy exponent (default=1.5) random between 0 and 2

Returns:
np.ndarray: Levy flight random numbers of shape (pop, m)

Lévy flight is employed to simulate the movement step size of
sea horses, which is conducive to the sea horse with the
high probability crossing to other positions in early iterations
and avoiding the excessive local exploitation of SHO
"""

def levy(pop, m, omega=1.5):
    # Calculate numerator and denominator for sigma
    num = gamma(1 + omega) * sin(pi * omega / 2)
    den = gamma((1 + omega) / 2) * omega * 2 ** ((omega - 1) / 2)
    
    # Standard deviation calculation
    sigma_u = (num / den) ** (1 / omega)
    
    # Generate random numbers
    u = np.random.normal(0, sigma_u, size=(pop, m))
    v = np.random.normal(0, 1, size=(pop, m))
    
    # Levy flight calculation
    z = u / (np.abs(v) ** (1 / omega))
    
    return  z  # Scale factor as in original MATLAB code