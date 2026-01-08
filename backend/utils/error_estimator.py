import numpy as np


def simple_estimation(surface_current_density):
    """
    Normalizes the surface current density to a range between 0 and 1.
    Values close to the minimum will be near 0, and values significantly 
    higher than the average (towards the maximum) will be near 1.
    """
    
    # Step 1: Get the magnitude (handle complex numbers if present)
    j_mag = np.abs(surface_current_density)
    
    # Step 2: Identify the range boundaries
    min_val = np.min(j_mag)
    max_val = np.max(j_mag)
    
    # Step 3: Linear remapping (Min-Max Scaling)
    # Formula: (x - min) / (max - min)
    range_width = max_val - min_val
    
    if range_width == 0:
        # Avoid division by zero if all values are identical
        return np.zeros_like(j_mag)
    
    # Values near min become 0, values near max become 1
    normalized_error = (j_mag - min_val) / range_width
        
    return normalized_error