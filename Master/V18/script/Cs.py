#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import numpy as np
from scipy.signal import find_peaks
import iminuit
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import norm
from typing import Tuple
from uncertainties import ufloat
from Linienbreite import fitmaker_2000
from Vollenergie import q_energy
from Kalibrierung import linear


matplotlib.rcParams.update({"font.size": 18})

# Einlesen der Caesium Messung
SKIP_ANFANG = 12
SKIP_ENDE = 14

caesium = pd.read_csv("./data/Cs1.Spe", skiprows=SKIP_ANFANG, header=None)
caesium = caesium.iloc[:-SKIP_ENDE]
caesium.columns = ["daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
caesium["daten"] = pd.to_numeric(caesium["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
caesium["index"] = caesium.index
untergrund["index"] = untergrund.index

# Untergrundmessung skalieren; Cs Messung dauerte 2865s; Untergrundmessung
# dauerte 76353s
untergrund["daten"] = untergrund["daten"] * (2865 / 76353)

# Untergrund entfernen
caesium["daten"] = caesium["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
caesium["daten"] = caesium["daten"].clip(lower=0)

# Peaks raussuchen um sie zu fitten
peaks_array, peaks_params = find_peaks(
    caesium["daten"], height=20, prominence=40, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Raus mit dem Müll
peaks = peaks.drop([0])
peaks = peaks.reset_index(drop=True)

# Schonmal neue Spalten anlegen
peaks["N"] = float(0)
peaks["N_err"] = float(0)

plt.figure(figsize=(21, 9), dpi=500)  # TODO andere dpi ausprobieren
plt.bar(
    caesium["index"],
    caesium["daten"],
    linewidth=2,
    width=1.1,
    label=r"$^{137}\mathrm{Cs}$",
    color="royalblue",
)
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="orange", label="Peaks")
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(caesium["daten"].min(), caesium["daten"].max(), 10))

plt.ylim(caesium["daten"].min() - 30)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Caesium-Peaks.pdf")
plt.clf()

matplotlib.rcParams.update({"font.size": 8})


def scaled_gauss_cdf_b(x, s, mu, sigma, b):
    """
    Eine skalierte kumulative Normalverteilungsfunktion,
    die an die Signale angepasst werden soll.

    Parameter:

    x: array-like, bin edges; muss len(n) + 1 sein (n: Anzahl bins)

    s: Skalierungsparameter der Funktion; ist auch der Integralwert

    mu: Erwartungswert der Normalverteilung

    sigma: Standardabweichung der Normalverteilung
    """
    return s * norm(mu, sigma).cdf(x) + b * x


grenzen = pd.DataFrame(data={"L": [150, 15], "R": [150, 15]})
zorder1 = [1, 2]
zorder2 = [2, 1]

# Das ist jetzt auch wieder arschcode aber wir wollen fertig werden
# Wenn jemand ganz viel Zeit hat wäre es viel sinnvoller die fitmaker_2000 Funktion so zu schreiben
# dass sie die Fitparameter einfach zurückgibt und die ganze Logik die das schreiben in ein Dataframe
# und speichern als csv übernimmt einfach aus der Funktion auslagert
# Das hier sind die Fitparameter die weiter unten bestimmt wurden
peaks["s"] = pd.NA
peaks["b"] = pd.NA
peaks.at[0, "s"] = ufloat(1275.237, 70.842)
peaks.at[1, "s"] = ufloat(9282.567, 98.438)
peaks.at[0, "b"] = ufloat(20.0, 0.457)
peaks.at[1, "b"] = ufloat(15.0, 0.165)

peaks["N"] = pd.NA
for i in range(len(peaks)):
    fitmaker_2000(
        caesium,
        peaks,
        grenzen["L"][i],
        grenzen["R"][i],
        i,
        scaled_gauss_cdf_b,
        zorder1=zorder1[i],
        zorder2=zorder2[i],
    )
    # Frag nicht
    peaks.at[i, "N"] = peaks.at[i, "s"]
    N = peaks.at[
        i, "N"
    ]  # nur benötigt weil fstrings wenig flexibel sind; ein "" innerhalb des {} macht schon alles kaputt
    print(f"N_{i+1} = {N:.5f}")

# Umrechnung der Kanäle in Energien wurde in Kalibrierung.py bestimmt
alpha = ufloat(0.217564, 0)
beta = ufloat(104.802701, 0.000292)
peaks["Energie"] = linear(peaks["peaks"], alpha, beta)

# Wurden durch Fit bestimmt, updaten wenn nötig;
# Sind die Parameter des Fits für Q(E)
a = ufloat(22.673, 5.031)
b = ufloat(-0.821, 0.033)
peaks["Q"] = q_energy(peaks["Energie"], a, b)
peaks.to_csv("./build/peaks_Cs.csv")
