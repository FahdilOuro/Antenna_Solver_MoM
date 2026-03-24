import math

# Constant for vacuum permeability (H/m) 
MU_0 = 1.257e-6     # Egale à 4π×10^−7 H/m

def calculate_normal_surface_impedance(frequency, conductivity, mu_r=1.0):
    """
    Calculates the surface impedance (Zs) for a normal conductor.
    
    Formula: Zs = (1 + j) * sqrt((omega * mu) / (2 * sigma))
    Where mu = mu_0 * mu_r
    
    Args:
        frequency (float): Signal frequency in Hertz (Hz).
        conductivity (float): Electrical conductivity in Siemens per meter (S/m).
        mu_r (float, optional): Relative permeability of the material. Defaults to 1.0.
        
    Returns:
        complex: The surface impedance Zs in Ohms.
    """
    # Total permeability is vacuum permeability times relative permeability
    # We use 1.0 as default because most conductors (Cu, Ag, Au) are non-magnetic
    mu = MU_0 * mu_r
    
    # Calculate angular frequency (omega = 2 * pi * f)
    omega = 2 * math.pi * frequency
    
    # Calculate the skin effect term: sqrt((omega * mu) / (2 * sigma))
    # This term represents both the real (Rs) and imaginary (Xs) parts
    term = math.sqrt((omega * mu) / (2 * conductivity))
    
    # Surface impedance Zs = Rs + jXs = (1 + j) * term
    zs = (1 + 1j) * term
    
    return zs

def calculate_superconductor_surface_impedance(frequency, london_depth, mu_r=1.0):
    """
    Calculates the surface impedance (Zs) for a superconductor.
    
    Formula: Zs = j * omega * mu * lambda_L
    Where mu = mu_0 * mu_r
    
    Args:
        frequency (float): Signal frequency in Hertz (Hz).
        london_depth (float): London penetration depth in meters (m).
        mu_r (float, optional): Relative permeability. Defaults to 1.0.
        
    Returns:
        complex: The surface impedance Zs in Ohms.
    """
    mu = MU_0 * mu_r
    omega = 2 * math.pi * frequency
    
    # For superconductors, the impedance is purely imaginary (inductive)
    # Zs = j * omega * mu * lambda_L
    zs = 1j * omega * mu * london_depth
    
    return zs