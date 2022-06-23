#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.constants as const
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import uncertainties.unumpy as unp


def magnetfeldstärke(I, N, R):
    """Berechnet die magnetische Feldstärke für eine gegebene Stromstärke."""
    B = const.mu_0 * (8 * I * N) / (np.sqrt(125) * R)
    return B


def gerade(x, m, b):
    """Gerade für die lineare Regression."""
    y = m * x + b
    return y


# Messwerte einlesen
df_messung = pd.read_csv("data/messwerte.txt", delimiter=" ")

# Frequenz kHz->Hz
df_messung["Frequenz[Hz]"] = df_messung["Frequenz[kHz]"] * 10**3

# Spulendaten
N_SWEEP = 11
N_HOR = 154
N_VERT = 20
R_SWEEP = 0.16390
R_HOR = 0.15790
R_VERT = 0.11735

# Magnetfeld der Sweep-Spule
df_messung["BSweepL[T]"] = magnetfeldstärke(
    df_messung["sweepPeakL[A]"], N_SWEEP, R_SWEEP
)
df_messung["BSweepR[T]"] = magnetfeldstärke(
    df_messung["sweepPeakR[A]"], N_SWEEP, R_SWEEP
)
# Magnetfeld der horizontalen Spule
df_messung["Bhor.L[T]"] = magnetfeldstärke(df_messung["hor.PeakL[A]"], N_HOR, R_HOR)
df_messung["Bhor.R[T]"] = magnetfeldstärke(df_messung["hor.PeakR[A]"], N_HOR, R_HOR)

# Gesamtfeld beider Spulen
df_messung["Bges.L[T]"] = df_messung["BSweepL[T]"] + df_messung["Bhor.L[T]"]
df_messung["Bges.R[T]"] = df_messung["BSweepR[T]"] + df_messung["Bhor.R[T]"]

print(df_messung)

# Regressionen berechnen
params87, cov87 = np.polyfit(
    df_messung["Frequenz[Hz]"], df_messung["Bges.L[T]"], deg=1, cov=True
)
errors87 = np.sqrt(np.diag(cov87))
params87_err = unp.uarray(params87, errors87)

params85, cov85 = np.polyfit(
    df_messung["Frequenz[Hz]"], df_messung["Bges.R[T]"], deg=1, cov=True
)
errors85 = np.sqrt(np.diag(cov85))
params85_err = unp.uarray(params85, errors85)

print(f"Parameter Regression 87 Rb: m = {params87_err[0]}, b = {params87_err[1]}")
print(f"Parameter Regression 85 Rb: m = {params85_err[0]}, b = {params85_err[1]}")

# Plot Frequenz (x) -> Magnetfeldstärke (y)
plt.plot(
    df_messung["Frequenz[Hz]"],
    df_messung["Bges.L[T]"],
    color="slateblue",
    label=r"$^{87} Rb$",
    marker="x",
    ls="",
)
plt.plot(
    df_messung["Frequenz[Hz]"],
    df_messung["Bges.R[T]"],
    color="cornflowerblue",
    label=r"$^{85} Rb$",
    marker="x",
    ls="",
)

# Regressionen
plt.plot(
    np.linspace(df_messung["Frequenz[Hz]"].min(), df_messung["Frequenz[Hz]"].max()),
    gerade(
        np.linspace(df_messung["Frequenz[Hz]"].min(), df_messung["Frequenz[Hz]"].max()),
        params87[0],
        params87[1],
    ),
    color="slateblue",
    label="Regression",
    ls="-",
)

plt.plot(
    np.linspace(df_messung["Frequenz[Hz]"].min(), df_messung["Frequenz[Hz]"].max()),
    gerade(
        np.linspace(df_messung["Frequenz[Hz]"].min(), df_messung["Frequenz[Hz]"].max()),
        params85[0],
        params85[1],
    ),
    color="cornflowerblue",
    label="Regression",
    ls="-",
)

plt.xlabel(r"$f [Hz]$")
plt.ylabel(r"$B [T]$")
plt.legend()
plt.tight_layout()
plt.savefig("build/plot1.pdf")

# Berechnnung der gyromagnetischen Faktoren
mub = 13996244936.1 #Hz T^-1
mub2 = 5.788381806*10**(-5) #eV T^-1
g87 = const.h/(params87_err[0]*mub)
g85 = const.h/(params85_err[0]*mub)

print(f"Gyromagnetischer Faktor 87 Rb: {g87}")
print(f"Gyromagnetischer Faktor 85 Rb: {g85}")
