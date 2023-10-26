#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Verwende LaTeX für Textrendering
rcParams['text.usetex'] = True

TEM00 = pd.read_csv("./data/TEM00.txt", sep="\s+", header=1)
TEM10 = pd.read_csv("./data/TEM10.txt", sep="\s+", header=1)

# def expTEM00

plt.plot(TEM00["r"], TEM00["I"], "x", label=(r"$\mathrm{TEM}_{00}$"))
plt.plot(TEM10["r"], TEM10["I"], "x", label=(r"$\mathrm{TEM}_{10}$"))
plt.xlabel(r"$r/\mathrm{cm}$")
plt.ylabel(r"$I/\mu \mathrm{A}$")
plt.legend()
plt.grid(True)
plt.savefig("./build/TEM-Moden.pdf")
plt.clf()
