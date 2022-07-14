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
mub = 9.2740100783*10**(-24) #J/T
g87 = const.h/(params87_err[0]*mub)
g85 = const.h/(params85_err[0]*mub)

print(f"Gyromagnetischer Faktor 87 Rb: {g87}")
print(f"Gyromagnetischer Faktor 85 Rb: {g85}")

# Berechnung der Kernspins
S = 0.5
L = 0
J = 0.5


def g_J(J, L, S):
    """Funktion für die Berechnung von g_J."""
    return 1+(J*(J+1) - L*(L+1) + S*(S+1))/(2*J*(J+1))


gJ = g_J(S, L, J)
I87 = 0.5 * (1 / g87 * gJ - 1)
I85 = 0.5 * (1 / g85 * gJ - 1)
print("Kernspin von Rb87:", I87)
print("Kernspin von Rb85:", I85)
I87_lit = 3/2
I85_lit = 5/2
AbwI87 = np.abs(I87-I87_lit)/I87_lit*100
AbwI85 = np.abs(I85-I85_lit)/I85_lit*100
print("Delta I87 =", AbwI87,"%")
print("Delta I85 =", AbwI85,"%")

# Isotopenateile
T87 = 524
T85 = 1037

A87 = (524+1037)/524
A85 = (524+1037)/1037

print(f"Anteil 87 Rb: {A87}")
print(f"Anteil 85 Rb: {A85}")
