#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares

MAKE_PLOT = True


# Einlesen der Kalibrationsmessung
SKIP_ANFANG = 12
SKIP_ENDE = 14

europium = pd.read_csv("./data/Eu1.Spe", skiprows=SKIP_ANFANG, header=None)
europium = europium.iloc[:-SKIP_ENDE]  # Entferne den Rotz am Ende
europium.columns = ["Daten"]
europium["index"] = europium.index

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
europium["data"] = pd.to_numeric(europium["Daten"], errors="coerce")

# Einlesen der Literaturwerte
europium_lit = pd.read_csv(
    "./data/LiteraturwerteEu.csv",
    skiprows=1,
    sep="\s+",
    header=None,
)
europium_lit.columns = ["Energie", "Unsicherheit(E)", "Intensität", "Unsicherheit(I)"]
europium_lit = europium_lit.drop([13])  # Ein Wert zuviel

# Peaks bestimmen und mit den zugehörigen Parametern in Dataframe speichern
peaks_array, peaks_params = find_peaks(
    europium["Daten"], height=20, prominence=20, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Peaks die eher dem Untergrundrauschen zuzuordnen sind entfernen
peaks = peaks.drop([0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16])

# Nach der Höhe des Peaks absteigend sortieren
peaks.sort_values(by="peak_heights", inplace=True, ascending=False)


def linear(K, alpha, beta):
    """Fit zwischen Kanalnummer und Energie"""
    return alpha * K + beta


least_squares = LeastSquares(
    peaks["peaks"], europium_lit["Energie"], europium_lit["Unsicherheit(E)"], linear
)
m = Minuit(least_squares, alpha=0, beta=0)
m.migrad()
m.hesse()

print(peaks.info())
print(peaks)

plt.errorbar(
    peaks["peaks"],
    europium_lit["Energie"],
    europium_lit["Unsicherheit(E)"],
    fmt="ok",
    label="data",
)
plt.plot(peaks["peaks"], linear(peaks["peaks"], *m.values), label="fit")

# display legend with some fit info
fit_info = [
    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info), frameon=False)
plt.xlabel(r"$\mathrm{Channels}$")
plt.ylabel(r"$\mathrm{Energy}/\mathrm{keV}$")
plt.savefig("./build/Europium-Fit.pdf")
plt.clf()
