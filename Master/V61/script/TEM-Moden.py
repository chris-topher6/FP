#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Verwende LaTeX für Textrendering
rcParams["text.usetex"] = True

TEM00 = pd.read_csv("./data/TEM00.txt", sep="\s+", header=1)
TEM10 = pd.read_csv("./data/TEM10.txt", sep="\s+", header=1)


def expTEM00(r, Imax, I0, r0, omega):
    I = Imax * np.exp((-((r - r0) ** 2)) / (2 * omega**2)) + I0
    return I


def expTEM10(r, Imax, I0, r0, omega):
    I = (
        Imax
        * ((4 * (r - r0) ** 2) / (omega**2))
        * np.exp((-((r - r0) ** 2)) / (2 * omega**2))
        + I0
    )
    return I


fit00_params, fit00_covariance = curve_fit(expTEM00, TEM00["r"], TEM00["I"])
fit10_params, fit10_covariance = curve_fit(expTEM10, TEM10["r"], TEM10["I"])
print(fit00_params)
print(fit10_params)


plt.plot(TEM00["r"], expTEM00(TEM00["r"], *fit00_params))
plt.plot(TEM10["r"], expTEM10(TEM10["r"], *fit10_params))
plt.plot(TEM00["r"], TEM00["I"], "x", label=(r"$\mathrm{TEM}_{00}$"))
plt.plot(TEM10["r"], TEM10["I"], "x", label=(r"$\mathrm{TEM}_{10}$"))
plt.xlabel(r"$r/\mathrm{cm}$")
plt.ylabel(r"$I/\mu \mathrm{A}$")
plt.legend()
plt.grid(True)
plt.savefig("./build/TEM-Moden.pdf")
plt.clf()
