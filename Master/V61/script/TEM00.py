#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Verwende LaTeX für Textrendering
rcParams['text.usetex'] = True

TEM00 = pd.read_csv("./data/TEM00.txt", sep="\s+", header=1)

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Fit durchführen
popt, _ = curve_fit(gauss, TEM00["r"], TEM00["I"])
# Extrahieren der Fit-Parameter
A_fit, mu_fit, sigma_fit = popt

x = np.linspace(-10, 25, 1000)

plt.plot(TEM00["r"], TEM00["I"], "x", label=(r"$\mathrm{TEM}_{00}$"))
plt.plot(x, gauss(x, A_fit, mu_fit, sigma_fit), label="Fit")
plt.xlabel(r"$r/\mathrm{cm}$")
plt.ylabel(r"$I/\mu \mathrm{A}$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./build/TEM00.pdf")
plt.clf()
