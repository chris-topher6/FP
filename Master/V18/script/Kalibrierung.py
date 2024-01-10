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

# Einlesen der Kalibrationsmessung
SKIP_ANFANG = 12
SKIP_ENDE = 14

europium = pd.read_csv("./data/Eu1.Spe", skiprows=SKIP_ANFANG, header=None)
europium = europium.iloc[:-SKIP_ENDE]  # Entferne den Rotz am Ende
europium.columns = ["Daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
europium["data"] = pd.to_numeric(europium["Daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
europium["index"] = europium.index
untergrund["index"] = untergrund.index

# Einlesen der Literaturwerte
europium_lit = pd.read_csv(
    "./data/LiteraturwerteEu.csv",
    skiprows=1,
    sep="\s+",
    header=None,
)
europium_lit.columns = ["Energie", "Unsicherheit(E)", "Intensität", "Unsicherheit(I)"]
europium_lit = europium_lit.drop([13])  # Ein Wert zuviel

# Untergrundmessung dauerte 76353s; Eu Messung dauerte 2804s; normiere auf
# letzteres für Plot beider Messungen
untergrund["daten"] = untergrund["daten"] * (2804 / 76353)

# Untergrund entfernen
europium["data"] = europium["data"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
europium["data"] = europium["data"].clip(lower=0)

# Peaks bestimmen und mit den zugehörigen Parametern in Dataframe speichern
peaks_array, peaks_params = find_peaks(
    europium["data"], height=15, prominence=19, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Peaks die eher dem Untergrundrauschen zuzuordnen sind entfernen
peaks = peaks.drop([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17])

# Plot der Kalibrationsmessung
plt.figure(figsize=(21, 9))  # TODO andere dpi ausprobieren

plt.bar(
    europium["index"],
    europium["data"],
    linewidth=2,
    width=1.1,
    label=r"$^{152}\mathrm{Eu}$",
    color="royalblue",
)
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="orange", label="Peaks")

plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(europium["data"].min(), europium["data"].max(), 10))

plt.ylim(europium["data"].min() - 30)

plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Europium-Peaks.pdf")
plt.clf()

# Nach der Kanalnummer aufsteigend sortieren
peaks.sort_values(by="peaks", inplace=True, ascending=True)
europium_lit.sort_values(by="Energie", inplace=True, ascending=True)

# Index beider Dfs zurücksetzen
peaks = peaks.reset_index(drop=True)
europium_lit = europium_lit.reset_index(drop=True)

# Nochmal alles in ein Df damit alles leichter gespeichert werden kann
peaks = pd.concat([europium_lit, peaks], axis=1)


def linear(K, alpha, beta):
    """Fit zwischen Kanalnummer und Energie"""
    return alpha * K + beta


least_squares = LeastSquares(
    peaks["peaks"], europium_lit["Energie"], europium_lit["Unsicherheit(E)"], linear
)
m = Minuit(least_squares, alpha=0, beta=0)
m.migrad()
m.hesse()

# print(least_squares.pulls(m.values)) So könnte man die Pulls ploten, sind hier aber gewaltig...

plt.errorbar(
    peaks["peaks"],
    europium_lit["Energie"],
    # xerr=np.sqrt(peaks["peak_heights"]),  # Poisson-Verteilt das ist aber quatsch,
    # die Kanäle haben keine Unsicherheit, nur die Höhe der Peaks hat eine
    yerr=europium_lit["Unsicherheit(E)"],
    fmt=".",
    label="data",
    color="royalblue",
)
plt.plot(peaks["peaks"], linear(peaks["peaks"], *m.values), label="fit", color="orange")

# display legend with some fit info
# fit_info = [
#     f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
# ]
fit_info = []  # Chi^2 sieht nicht gut aus

with open("./build/Fitparameter_Kalib.txt", "w") as file:
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.6f} \\pm {e:.6f}$")
        file.write(f"{p} = ${v:.6f} \\pm {e:.6f}$\n")

# Für Weiterbenutzung in anderen Skripten
peaks.to_csv("./build/peaks.csv", index=False)

plt.legend(title="\n".join(fit_info), frameon=False)
plt.xlabel(r"$\mathrm{Channels}$")
plt.ylabel(r"$\mathrm{Energy}/\mathrm{keV}$")
plt.tight_layout()
plt.savefig("./build/Europium-Fit.pdf")
plt.clf()
