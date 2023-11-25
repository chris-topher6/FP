#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares

MAKE_PLOT = False


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

peaks, _ = find_peaks(europium["Daten"], height=20, prominence=20, distance=10)
peaks = pd.Series(peaks)

# Peaks die eher dem Untergrundrauschen zuzuordnen sind entfernen
peaks = peaks.drop([0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16])


def linear(K, alpha, beta):
    return alpha * K + beta


least_squares = LeastSquares(
    peaks, europium_lit["Energie"], europium_lit["Unsicherheit(E)"], linear
)
m = Minuit(least_squares, alpha=0, beta=0)
m.migrad()
m.hesse()

print(len(peaks))
print(peaks.info())
print(europium.info())
print(len(europium_lit))
print(europium_lit.info())
print(europium_lit)


if MAKE_PLOT == True:
    # Plot der Kalibrationsmessung
    plt.figure(figsize=(21, 9))

    plt.bar(europium["index"], europium["Daten"], linewidth=2, width=1.1)
    plt.plot(peaks, europium["Daten"][peaks], "x", color="orange")

    plt.xticks(np.linspace(0, 8191, 10))
    plt.yticks(np.linspace(europium["Daten"].min(), europium["Daten"].max(), 10))

    plt.ylim(europium["Daten"].min() - 30)

    plt.xlabel(r"Kanäle")
    plt.ylabel(r"Signale")

    plt.grid(True, linewidth=0.1)

    plt.savefig("./build/Europium-Test.pdf")
