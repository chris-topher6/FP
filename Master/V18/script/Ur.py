#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks
from uncertainties import ufloat
from Linienbreite import fitmaker_2000
from Vollenergie import q_energy
from Kalibrierung import linear
from Cs import scaled_gauss_cdf_b

matplotlib.rcParams.update({"font.size": 18})

# Das hier sind die Fitparameter aus der Kalibrierung
alpha = ufloat(0.207452, 0)
beta = ufloat(-1.622490, 0.000381)

# Einlesen der Uran-Messung
SKIP_ANFANG = 12
SKIP_ENDE = 14

uran = pd.read_csv("./data/U1.Spe", skiprows=SKIP_ANFANG, header=None)
uran = uran.iloc[:-SKIP_ENDE]
uran.columns = ["daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
uran["daten"] = pd.to_numeric(uran["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
uran["index"] = uran.index
untergrund["index"] = untergrund.index

# Peaks aus dem Untergrund raussuchen um sie vergleichen zu können
peaks_array_untergrund, peaks_params_untergrund = find_peaks(
    untergrund["daten"], height=20, prominence=20, distance=15
)
peaks_untergrund = pd.DataFrame(peaks_params_untergrund)
peaks_untergrund["peaks"] = peaks_array_untergrund

# Aufräumen
indizes_zum_behalten = [0, 14, 18, 39, 50, 63, 71, 90, 95, 97, 99, 105, 106, 108, 109]
# peaks_untergrund_kurz = peaks_untergrund.iloc[indizes_zum_behalten]
peaks_untergrund = peaks_untergrund.iloc[indizes_zum_behalten]

print("Peaks Im Untergrund")
print(peaks_untergrund["peaks"])
print("---")
# Erster Plot der Untergrundmessung
plt.figure(figsize=(21, 9), dpi=500)
plt.bar(
    untergrund["index"],
    untergrund["daten"],
    linewidth=2,
    width=1.1,
    label="Background",
    color="royalblue",
)
plt.plot(
    peaks_untergrund["peaks"],
    peaks_untergrund["peak_heights"],
    "x",
    color="orange",
    label="Peaks",
)
# plt.plot(
#     peaks_untergrund_kurz["peaks"],
#     peaks_untergrund_kurz["peak_heights"],
#     "x",
#     color="green",
#     label="Peaks",
# )
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(untergrund["daten"].min(), untergrund["daten"].max(), 10))

plt.ylim(untergrund["daten"].min() - 10)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Untergrund-Peaks-unskaliert.pdf")
plt.clf()

# Untergrundmessung skalieren; Uran-Messung dauerte 2968s; Untergrundmessung
# dauerte 76353s
untergrund["daten"] = untergrund["daten"] * (2968 / 76353)

# Untergrund entfernen
uran["daten"] = uran["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
uran["daten"] = uran["daten"].clip(lower=0)

# Peaks raussuchen um sie zu fitten
peaks_array, peaks_params = find_peaks(
    uran["daten"], height=20, prominence=60, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Aufräumen
indizes_zum_behalten = [
    0,
    11,
    16,
    20,
    22,
    27,
    31,
    32,
    38,
    44,
    45,
    47,
    48,
    50,
    51,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
]
# peaks_uran_kurz = peaks.iloc[indizes_zum_behalten]
peaks_uran = peaks.iloc[indizes_zum_behalten]

plt.figure(figsize=(21, 9), dpi=500)
plt.bar(
    uran["index"],
    uran["daten"],
    linewidth=2,
    width=1.1,
    label="Uranium",
    color="royalblue",
)
plt.plot(
    peaks_uran["peaks"], peaks_uran["peak_heights"], "x", color="orange", label="Peaks"
)

# plt.plot(
#     peaks_uran_kurz["peaks"],
#     peaks_uran_kurz["peak_heights"],
#     "x",
#     color="green",
#     label="Peaks",
# )
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(uran["daten"].min(), uran["daten"].max(), 10))

plt.ylim(uran["daten"].min() - 10)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Uran-Peaks.pdf")
plt.clf()
plt.close()

peaks_uran = peaks_uran.reset_index(drop=True)

# Jetzt nur noch die Peaks die nicht im Untergrund auftreten
indizes_zum_behalten = [1, 3, 5, 6, 7, 9, 11, 12, 17, 18, 19, 21, 22]
peaks_uran_uran = peaks_uran.iloc[indizes_zum_behalten]

print("Peaks im Uran")
print(peaks_uran["peaks"])

plt.figure(figsize=(21, 9), dpi=500)
plt.bar(
    uran["index"],
    uran["daten"],
    linewidth=2,
    width=1.1,
    label="Uranium",
    color="royalblue",
)
plt.plot(
    peaks_uran["peaks"],
    peaks_uran["peak_heights"],
    "x",
    color="gray",
    label="Background Peaks",
)

plt.plot(
    peaks_uran_uran["peaks"],
    peaks_uran_uran["peak_heights"],
    "x",
    color="orange",
    label="Uranium Peaks",
)
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(uran["daten"].min(), uran["daten"].max(), 10))

plt.ylim(uran["daten"].min() - 10)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Uran-Uran-Peaks.pdf")
plt.clf()
plt.close()

peaks_uran_uran = peaks_uran_uran.reset_index(drop=True)

# Schonmal neue Spalten anlegen
peaks_uran_uran["N"] = float(0)
peaks_uran_uran["N_err"] = float(0)

# Kleinformatigere Abbildungen ab hier
matplotlib.rcParams.update({"font.size": 8})

grenzen = pd.DataFrame(
    data={
        "L": [6, 8, 5, 8, 4, 8, 4, 4, 15, 11, 15, 15, 15],
        "R": [7, 6, 4, 5, 3, 8, 5, 2, 15, 15, 15, 15, 15],
    }
)

for i in range(len(peaks_uran_uran)):
    fitmaker_2000(
        uran,
        peaks_uran_uran,
        grenzen["L"][i],
        grenzen["R"][i],
        i,
        scaled_gauss_cdf_b,
        "uranium",
        zorder1=1,
        zorder2=2,
    )
