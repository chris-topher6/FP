#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Verwende LaTeX für Textrendering
rcParams["text.usetex"] = True

# Daten einlesen
Pol = pd.read_csv("./data/Intensität-Winkel-Messung.txt", sep="\s+", header=1)
Pol["Fehler"] = 0.04  # Schätzung


def I(alpha, Imax, alpha0, I0):
    """Funktion für den Fit der Winkelabhängigkeit der Intensität"""
    I = Imax * (np.sin(np.radians(alpha) + np.radians(alpha0))) ** 2 + I0
    return I


# Fit durchführen
# Startwerte
initial_params = [0.86, 160, 0]
fit_params, fit_covariance = curve_fit(I, Pol["a"], Pol["I"], p0=initial_params)

# Linspace zum Plotten
x = np.linspace(0, 350, 1000)

# Pulls berechnen
fitwerte = I(Pol["a"], *fit_params)
pulls = (Pol["I"] - fitwerte) / Pol["Fehler"]

# Coole Plots
fig, axs = plt.subplots(
    2, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1]}
)
axs[0].errorbar(
    Pol["a"],
    Pol["I"],
    yerr=Pol["Fehler"],
    fmt="x",
    label=(r"$\mathrm{Messwerte}$"),
    color="royalblue",
)  # Fehler
axs[0].plot(
    x, I(x, *fit_params), color="lightsteelblue", label=(r"$\mathrm{Fit}$")
)  # Fit
# axs00[0].plot(TEM00["r"], TEM00["I"], "x", label=(r"$\mathrm{TEM}_{00}$")) # Daten
axs[0].legend()
axs[0].set_ylabel(r"$I/ \mathrm{mW}$")
axs[0].grid(True)

axs[1].bar(Pol["a"], pulls, width=7.5, color="lightsteelblue")
axs[1].axhline(0, color="orangered", linewidth=0.8)  # Linie bei Pull 0
axs[1].set_xlabel(r"$a/^{\circ}$")
axs[1].set_ylabel(r"$\mathrm{Pull}/\Delta I$")
axs[1].set_yticks(np.linspace(-30, 30, 5))
axs[1].grid(True)

plt.tight_layout()
plt.savefig("./build/Polarisation.pdf")
plt.clf()

# Parameter des Fits speichern

# Extrahiere Fehler aus der Kovarianzmatrix
fit_errors = np.sqrt(np.diag(fit_covariance))

# Schreibe in Datei
with open("./build/fit_parameters_polarisation.txt", "w") as file:
    file.write("Fitparameter für die Polarisationsmessung:\n")
    file.write(f"Imax: {fit_params[0]} ± {fit_errors[0]}\n")
    file.write(f"alpha_0: {fit_params[1]} ± {fit_errors[1]}\n")
    file.write(f"I_0: {fit_params[2]} ± {fit_errors[2]}\n")

    file.write("\nGerundet auf zwei Nachkommastellen ergibt sich:\n")
    file.write(f"Imax: {fit_params[0]:.2f} ± {fit_errors[0]:.2f}\n")
    file.write(f"alpha_0: {fit_params[1]:.2f} ± {fit_errors[1]:.2f}\n")
    file.write(f"I_0: {fit_params[2]:.2f} ± {fit_errors[2]:.2f}\n")
