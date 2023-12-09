#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares


def aktivitätsgesetzt(t, A0, tau):
    A = A0 * np.exp(((-np.log(2)) / tau) * t)
    return A
