#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import iminuit
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import norm

matplotlib.rcParams.update({"font.size": 18})

SKIP_ANFANG = 12
SKIP_ENDE = 14

# Einlesen der Kalibrations- und Untergrundmessung
europium = pd.read_csv("./data/Eu1.Spe", skiprows=SKIP_ANFANG, header=None)
europium.columns = ["daten"]
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]

# Rotz am Ende entfernen
europium = europium.iloc[:-SKIP_ENDE]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
europium["daten"] = pd.to_numeric(europium["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index hinzufügen
europium["index"] = europium.index
untergrund["index"] = untergrund.index

# Plot der Untergrundmessung
plt.figure(figsize=(21, 9))

plt.bar(
    untergrund["index"],
    untergrund["daten"],
    linewidth=2,
    width=1.1,
    color="orange",
    label="Background",
)

plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(europium["daten"].min(), europium["daten"].max(), 10))

plt.ylim(europium["daten"].min() - 30)

plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Untergrund.pdf")
plt.clf()

# Untergrundmessung dauerte 76353s; Eu Messung dauerte 2804s; normiere auf
# letzteres für Plot beider Messungen
untergrund["daten"] = untergrund["daten"] * (2804 / 76353)

# Plot der beiden Messungen in einem
plt.figure(figsize=(21, 9))

plt.bar(
    europium["index"],
    europium["daten"],
    linewidth=2,
    width=1.1,
    label=r"$^{152}\mathrm{Eu}$",
)
plt.bar(
    untergrund["index"],
    untergrund["daten"],
    linewidth=2,
    width=1.1,
    color="orange",
    label="Background",
)

plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(europium["daten"].min(), europium["daten"].max(), 10))

plt.ylim(europium["daten"].min() - 30)

plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Europium-Untergrund.pdf")
plt.clf()

# Untergrund entfernen
europium["daten"] = europium["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
europium["daten"] = europium["daten"].clip(lower=0)

# Vorher bestimmte Peaks einlesen
peaks = pd.read_csv("./build/peaks.csv")


def scaled_gauss_cdf(x, s, mu, sigma):
    """
    Eine skalierte kumulative Normalverteilungsfunktion,
    die an die Signale angepasst werden soll.

    Parameter:
    x: array-like, bin edges; muss len(n) + 1 sein (n: Anzahl bins)
    s: Skalierungsparameter der Funktion
    mu: Erwartungswert der Normalverteilung
    sigma: Standardabweichung der Normalverteilung
    """
    return s * norm(mu, sigma).cdf(x)


for i in range(len(peaks)):
    # Bereich um den jeweiligen Peak herum ausschneiden
    maske = (europium["index"] >= peaks["peaks"][i] - 50) & (
        europium["index"] <= peaks["peaks"][i] + 50
    )
    europium_cut = europium[maske]

    # Erstelle Bin-Kanten für den abgeschnittenen Datensatz
    cut_bin_edges = np.arange(
        europium_cut["index"].min(), europium_cut["index"].max() + 2
    )

    cost = ExtendedBinnedNLL(europium_cut["daten"], cut_bin_edges, scaled_gauss_cdf)

    # Schätze s über die maximale Höhe des Peaks
    peak_height = europium_cut["daten"].max()

    # Als Startwert für mu: Position des Peaks
    peak_position = peaks["peaks"][i]

    # Schätze die Breite des Peaks (sigma)
    peak_width = (europium_cut["index"].max() - europium_cut["index"].min()) * 0.1

    m = Minuit(cost, s=peak_height, mu=peak_position, sigma=peak_width)
    m.limits["s", "mu", "sigma"] = (0, 10000)
    m.migrad()
    m.hesse()

    print(m.values)
    print(m.errors)

    # Plot des Fits als Stufenfunktion
    plt.bar(
        europium_cut["index"],
        europium_cut["daten"],
        linewidth=2,
        width=1.1,
        color="blue",
        label=r"$^{152}\mathrm{Eu}$",
    )
    plt.stairs(
        np.diff(scaled_gauss_cdf(cut_bin_edges, *m.values)),
        # * europium_cut["daten"].sum(),
        cut_bin_edges,
        label="fit (steps)",
        color="orange",
        linewidth=2.2,
    )
    plt.legend()
    plt.xlabel(r"$\mathrm{Channels}$")
    plt.ylabel(r"$\mathrm{Energy}/\mathrm{keV}$")
    plt.tight_layout()
    plt.savefig(f"./build/Europium-Fit-Peak{i}.pdf")
    plt.clf()
