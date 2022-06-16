#!/usr/bin/env python3
import numpy as np
from scipy.constants as const


def magnetfeldstärke(I, N, R):
    """Berechnet die magnetische Feldstärke für eine gegebene Stromstärke."""
    B = const.mu_0*(8*I*N)/(np.sqrt(125)*R)
    return(B)
