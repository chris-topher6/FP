#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import find_peaks
from uncertainties import ufloat
from Linienbreite import fitmaker_2000
from Kalibrierung import linear
from Vollenergie import q_energy
from Cs import scaled_gauss_cdf_b
from Co import aktivität_q

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

# print("Peaks Im Untergrund")
# print(peaks_untergrund["peaks"])
# print("---")
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

# Auch in der skalierten Variante die Peaks finden
peaks_array_untergrund_skaliert, peaks_params_untergrund_skaliert = find_peaks(
    untergrund["daten"], height=0, prominence=1, distance=15
)
peaks_untergrund_skaliert = pd.DataFrame(peaks_params_untergrund_skaliert)
peaks_untergrund_skaliert["peaks"] = peaks_array_untergrund_skaliert

# Aufräumen
indizes_zum_behalten = [0, 13, 17, 39, 49, 54, 64, 74, 75, 76, 77, 78, 79]
# peaks_untergrund_kurz_skaliert = peaks_untergrund_skaliert.iloc[indizes_zum_behalten]
peaks_untergrund_skaliert = peaks_untergrund_skaliert.iloc[indizes_zum_behalten]

# Zweiter Plot der Untergrundmessung
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
    peaks_untergrund_skaliert["peaks"],
    peaks_untergrund_skaliert["peak_heights"],
    "x",
    color="orange",
    label="Peaks",
)
# plt.plot(
#     peaks_untergrund_kurz_skaliert["peaks"],
#     peaks_untergrund_kurz_skaliert["peak_heights"],
#     "x",
#     color="green",
#     label="Peaks",
# )
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(untergrund["daten"].min(), untergrund["daten"].max(), 10))

plt.ylim(untergrund["daten"].min() - 1)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Untergrund-Peaks-skaliert.pdf")
plt.clf()

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
indizes_zum_behalten = [1, 3, 5, 6, 7, 9, 10, 11, 12, 17, 18, 19, 21, 22]
peaks_uran_uran = peaks_uran.iloc[indizes_zum_behalten]

# print("Peaks im Uran")
# print(peaks_uran["peaks"])
# print("Peaks im Dataframe Uran Uran:")
# print(peaks_uran_uran["peaks"])

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
        "L": [6, 8, 5, 8, 4, 8, 7, 4, 2, 15, 11, 15, 15, 15],
        "R": [7, 6, 4, 5, 3, 8, 6, 5, 2, 15, 15, 15, 15, 15],
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

# Ausgabe der Energien bei der die Peaks liegen
peaks_keV = pd.DataFrame(data={"peaks": linear(peaks_uran_uran["peaks"], alpha, beta)})

# Die Konstanten werden benötigt
a = 8.91  # cm
r = 2.25  # cm
omega_4pi = 1 / 2 * (1 - a / (np.sqrt(a**2 + r**2)))

# Fitparamter für Q(E);
a_q = ufloat(4.757, 0.347)
b_q = ufloat(-0.915, 0.013)

peaks_keV["Q"] = ufloat(0, 0)
peaks_keV["A"] = ufloat(0, 0)
peaks_keV["I"] = [
    float(0.49),
    float(0.271),
    float(0.313),
    float(0.00046),
    float(0.205),
    float(1),  # ist natürlich quatsch, aber irgendeinen Wert muss man nehmen
    float(0.0117),
    float(0.5318),
    float(0.0161),
    float(4.892),
    float(1.262),
    float(3.10),
    float(5.831),
    float(3.968),
]
# ja ich weiß
peaks_keV["s"] = pd.Series(
    data=[
        ufloat(11833.671, 215.914),
        ufloat(15737.739, 175.705),
        ufloat(11680.016, 1387.264),
        ufloat(10920.247, 366.263),
        ufloat(9042.962, 1500.466),
        ufloat(5842.050, 217.646),
        ufloat(15338.125, 132.976),
        ufloat(4133.150, 597.108),
        ufloat(1702.320, 1.970),
        ufloat(3708.917, 0.153),
        ufloat(2634.706, 58.979),
        ufloat(748.940, 39.955),
        ufloat(1280.150, 46.168),
        ufloat(1599.454, 48.102),
        ufloat(987.948, 40.773),
    ]
)

# Messzeit
t = 2968
# print(peaks_keV)

for i in range(len(peaks_keV)):
    peaks_keV.at[i, "Q"] = q_energy(peaks_keV.at[i, "peaks"], a_q, b_q)
    peaks_keV.at[i, "A"] = aktivität_q(
        omega_4pi, peaks_keV.at[i, "s"], peaks_keV.at[i, "Q"], peaks_keV.at[i, "I"], t
    )
    Ai = peaks_keV.at[i, "A"]
    print(f"Die Aktivität von Peak {i} beträgt {Ai:.4f} bq")

plt.close()
peaks_keV.to_csv("./build/peaks_ur.csv")
