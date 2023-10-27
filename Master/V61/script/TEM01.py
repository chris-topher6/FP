#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.special import hermite

# Verwende LaTeX für Textrendering
rcParams['text.usetex'] = True

TEM01 = pd.read_csv("./data/TEM10.txt", sep="\s+", header=1)

#def gauss_with_hermite(x, a, b, A, mu, sigma):
#    hermite_1 = -a*x**2+b
#    return A * hermite_1 * np.exp(-(x - mu)**2 / (2 * sigma**2))

def gauss_with_hermite(x, A, mu, sigma):
    return A * 8*(x-mu)**2/(sigma**2) * np.exp(-2*(x - mu)**2/(2*sigma**2))

# Fit durchführen
popt, _ = curve_fit(gauss_with_hermite, TEM01["r"], TEM01["I"], p0=[1, 6.5, 6.8])
# Extrahieren der Fit-Parameter
A_fit, mu_fit, sigma_fit = popt

x = np.linspace(-10, 25, 1000)

plt.plot(TEM01["r"], TEM01["I"], "x", label=(r"$\mathrm{TEM}_{01}$"))
#plt.plot(x, gauss_with_hermite(x, 1, 6.5, 6.8), label="Chrisfit")
plt.plot(x, gauss_with_hermite(x, A_fit, mu_fit, sigma_fit), label="Curvefit")
plt.xlabel(r"$r/\mathrm{cm}$")
plt.ylabel(r"$I/\mu \mathrm{A}$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./build/TEM01.pdf")
plt.clf()
