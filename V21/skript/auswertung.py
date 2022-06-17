#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.constants as const
import matplotlib.pyplot as plt


def magnetfeldstärke(I, N, R):
    """Berechnet die magnetische Feldstärke für eine gegebene Stromstärke."""
    B = const.mu_0 * (8 * I * N) / (np.sqrt(125) * R)
    return B


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

plt.xlabel(r"$f [Hz]$")
plt.ylabel(r"$B [T]$")
plt.legend()
plt.tight_layout()
plt.savefig("build/plot1.pdf")
