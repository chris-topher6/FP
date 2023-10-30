#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Verwende LaTeX für Textrendering
rcParams["text.usetex"] = True

# Daten einlesen und Fehler berechnen (unter der Annahme, dass die Daten Poisson-verteilt sind)
TEM00 = pd.read_csv("./data/TEM00.txt", sep="\s+", header=1)
TEM00["Fehler"] = 0.09  # Schätzung
TEM10 = pd.read_csv("./data/TEM10.txt", sep="\s+", header=1)
TEM10["Fehler"] = 0.09  # Schätzung


def expTEM00(r, Imax, r0, omega):
    """Funktion für den Fit der TEM_00 Mode"""
    I = Imax * np.exp((-((r - r0) ** 2)) / (2 * omega**2)) + 0.0001
    return I


def expTEM10(r, Imax, r0, omega):
    """Funktion für den Fit der TEM_10 Mode"""
    I = (
        Imax
        * ((4 * (r - r0) ** 2) / (omega**2))
        * np.exp((-((r - r0) ** 2)) / (2 * omega**2))
        + 0.0001
    )
    return I


# Fit durchführen
fit00_params, fit00_covariance = curve_fit(expTEM00, TEM00["r"], TEM00["I"])
# Startwerte, Mittelpunkt scheint in etwa bei r0=5 zu liegen
initial_params01 = [1, 5, 1]
fit10_params, fit10_covariance = curve_fit(
    expTEM10, TEM10["r"], TEM10["I"], p0=initial_params01
)

# Linspace zum Plotten
x = np.linspace(-10, 25, 1000)

# Pulls berechnen
fitwerte00 = expTEM00(TEM00["r"], *fit00_params)
fitwerte10 = expTEM10(TEM10["r"], *fit10_params)

pulls00 = (TEM00["I"] - fitwerte00) / TEM00["Fehler"]
pulls10 = (TEM10["I"] - fitwerte10) / TEM10["Fehler"]

# Basic Plots
plt.plot(x, expTEM00(x, *fit00_params))
plt.plot(x, expTEM10(x, *fit10_params))
plt.plot(TEM00["r"], TEM00["I"], "x", label=(r"$\mathrm{TEM}_{00}$"))
plt.plot(TEM10["r"], TEM10["I"], "x", label=(r"$\mathrm{TEM}_{10}$"))
plt.xlabel(r"$r/\mathrm{cm}$")
plt.ylabel(r"$I/\mu \mathrm{A}$")
plt.legend()
plt.grid(True)
plt.savefig("./build/TEM-Moden.pdf")
plt.clf()

# Coole Plots
fig00, axs00 = plt.subplots(
    2, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1]}
)
axs00[0].errorbar(
    TEM00["r"],
    TEM00["I"],
    yerr=TEM00["Fehler"],
    fmt="x",
    label=(r"$\mathrm{TEM}_{00}$"),
    color="royalblue",
)  # Fehler
axs00[0].plot(
    x, expTEM00(x, *fit00_params), color="lightsteelblue", label=(r"$\mathrm{Fit}$")
)  # Fit
# axs00[0].plot(TEM00["r"], TEM00["I"], "x", label=(r"$\mathrm{TEM}_{00}$")) # Daten
axs00[0].legend()
axs00[0].set_ylabel(r"$I/\mu \mathrm{A}$")
axs00[0].grid(True)

axs00[1].bar(TEM00["r"], pulls00, width=1.5, color="lightsteelblue")
axs00[1].axhline(0, color="orangered", linewidth=0.8)  # Linie bei Pull 0
axs00[1].set_xlabel(r"$r/\mathrm{cm}$")
axs00[1].set_ylabel(r"$\mathrm{Pull}/\sigma_{I}$")
axs00[1].set_yticks(np.linspace(-10, 10, 5))
axs00[1].grid(True)

plt.tight_layout()
plt.savefig("./build/TEM00-Mode+Pull.pdf")
plt.clf()

fig10, axs10 = plt.subplots(
    2, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1]}
)
axs10[0].errorbar(
    TEM10["r"],
    TEM10["I"],
    yerr=TEM10["Fehler"],
    fmt="x",
    label=(r"$\mathrm{TEM}_{10}$"),
    color="royalblue",
)  # Fehler
axs10[0].plot(
    x, expTEM10(x, *fit10_params), color="lightsteelblue", label=(r"$\mathrm{Fit}$")
)  # Fit
# axs10[0].plot(TEM00["r"], TEM00["I"], "x", label=(r"$\mathrm{TEM}_{00}$")) # Daten
axs10[0].legend()
axs10[0].set_ylabel(r"$I/\mu \mathrm{A}$")
axs10[0].grid(True)

axs10[1].bar(TEM10["r"], pulls10, width=1.5, color="lightsteelblue")
axs10[1].axhline(0, color="orangered", linewidth=0.8)  # Linie bei Pull 0
axs10[1].set_xlabel(r"$r/\mathrm{cm}$")
axs10[1].set_ylabel(r"$\mathrm{Pull}/\sigma_{I}$")
axs10[1].set_yticks(np.linspace(-5, 5, 5))
axs10[1].grid(True)

plt.tight_layout()
plt.savefig("./build/TEM10-Mode+Pull.pdf")
plt.clf()

# Parameter des Fits speichern

# Extrahiere Fehler aus der Kovarianzmatrix
fit00_errors = np.sqrt(np.diag(fit00_covariance))
fit10_errors = np.sqrt(np.diag(fit10_covariance))

# Schreibe in Datei
with open("./build/fit_parameters.txt", "w") as file:
    file.write("Fitparameter für TEM00:\n")
    file.write(f"Imax: {fit00_params[0]} ± {fit00_errors[0]}\n")
    file.write(f"r0: {fit00_params[1]} ± {fit00_errors[1]}\n")
    file.write(f"omega: {fit00_params[2]} ± {fit00_errors[2]}\n")

    file.write("\nFitparameter für TEM10:\n")
    file.write(f"Imax: {fit10_params[0]} ± {fit10_errors[0]}\n")
    file.write(f"r0: {fit10_params[1]} ± {fit10_errors[1]}\n")
    file.write(f"omega: {fit10_params[2]} ± {fit10_errors[2]}\n")

    file.write("\nGerundet auf zwei Nachkommastellen ergibt sich:\n")
    file.write("Fitparameter für TEM00:\n")
    file.write(f"Imax: {fit00_params[0]:.2f} ± {fit00_errors[0]:.2f}\n")
    file.write(f"r0: {fit00_params[1]:.2f} ± {fit00_errors[1]:.2f}\n")
    file.write(f"omega: {fit00_params[2]:.2f} ± {fit00_errors[2]:.2f}\n")

    file.write("\nFitparameter für TEM10:\n")
    file.write(f"Imax: {fit10_params[0]:.2f} ± {fit10_errors[0]:.2f}\n")
    file.write(f"r0: {fit10_params[1]:.2f} ± {fit10_errors[1]:.2f}\n")
    file.write(f"omega: {fit10_params[2]:.2f} ± {fit10_errors[2]:.2f}\n")
