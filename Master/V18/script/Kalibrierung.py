#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares

matplotlib.rcParams.update({"font.size": 18})

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
    europium["data"], height=20, prominence=20, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Peaks die eher dem Untergrundrauschen zuzuordnen sind entfernen
peaks = peaks.drop([0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16])

if MAKE_PLOT == True:
    # Plot der Kalibrationsmessung
    plt.figure(figsize=(21, 9))

    plt.bar(europium["index"], europium["data"], linewidth=2, width=1.1)
    plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="orange")

    plt.xticks(np.linspace(0, 8191, 10))
    plt.yticks(np.linspace(europium["data"].min(), europium["data"].max(), 10))

    plt.ylim(europium["data"].min() - 30)

    plt.xlabel(r"Kanäle")
    plt.ylabel(r"Signale")

    plt.grid(True, linewidth=0.1)
    plt.tight_layout()
    plt.savefig("./build/Europium-Peaks.pdf")
    plt.clf()

# Nach der Kanalnummer aufsteigend sortieren
peaks.sort_values(by="peaks", inplace=True, ascending=True)
europium_lit.sort_values(by="Energie", inplace=True, ascending=True)

print(peaks["peaks"])
print(europium_lit["Energie"])


def linear(K, alpha, beta):
    """Fit zwischen Kanalnummer und Energie"""
    return alpha * K + beta


least_squares = LeastSquares(
    peaks["peaks"], europium_lit["Energie"], europium_lit["Unsicherheit(E)"], linear
)
m = Minuit(least_squares, alpha=0, beta=0)
m.migrad()
m.hesse()

# print(peaks.info())
# print(peaks)
# print(least_squares.pulls(m.values)) So könnte man die Pulls ploten, sind hier aber gewaltig...

plt.errorbar(
    peaks["peaks"],
    europium_lit["Energie"],
    # xerr=np.sqrt(peaks["peak_heights"]),  # Poisson-Verteilt das ist aber quatsch,
    # die Kanäle haben keine Unsicherheit, nur die Höhe der Peaks hat eine
    yerr=europium_lit["Unsicherheit(E)"],
    fmt=".",
    label="data",
)
plt.plot(peaks["peaks"], linear(peaks["peaks"], *m.values), label="fit")

# display legend with some fit info
# fit_info = [
#     f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
# ]
fit_info = []  # Chi^2 sieht nicht gut aus

with open("./build/Fitparameter_Kalib.txt", "w") as file:
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.6f} \\pm {e:.6f}$")
        file.write(f"{p} = ${v:.6f} \\pm {e:.6f}$\n")

plt.legend(title="\n".join(fit_info), frameon=False)
plt.xlabel(r"$\mathrm{Channels}$")
plt.ylabel(r"$\mathrm{Energy}/\mathrm{keV}$")
plt.tight_layout()
plt.savefig("./build/Europium-Fit.pdf")
plt.clf()
