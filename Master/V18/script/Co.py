#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from scipy.stats import norm
from scipy.integrate import quad
import scipy
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares
from typing import Tuple
from uncertainties import ufloat
from Linienbreite import fitmaker_2000
from Vollenergie import q_energy
from Kalibrierung import linear, linear_invers
from Cs import scaled_gauss_cdf_b, scaled_gauss_pdf_b

matplotlib.rcParams.update({"font.size": 18})

# Das hier sind die Fitparameter aus der Kalibrierung
alpha = ufloat(0.207452, 0)
beta = ufloat(-1.622490, 0.000381)

# Einlesen der Cobalt Messung
SKIP_ANFANG = 12
SKIP_ENDE = 14

cobalt = pd.read_csv("./data/Co1.Spe", skiprows=SKIP_ANFANG, header=None)
cobalt = cobalt.iloc[:-SKIP_ENDE]
cobalt.columns = ["daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
cobalt["daten"] = pd.to_numeric(cobalt["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
cobalt["index"] = cobalt.index
untergrund["index"] = untergrund.index

# Untergrundmessung skalieren; Co Messung dauerte 4021s; Untergrundmessung
# dauerte 76353s
untergrund["daten"] = untergrund["daten"] * (4021 / 76353)

# Untergrund entfernen
cobalt["daten"] = cobalt["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
cobalt["daten"] = cobalt["daten"].clip(lower=0)

# Peaks raussuchen um sie zu fitten
peaks_array, peaks_params = find_peaks(
    cobalt["daten"], height=20, prominence=40, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

plt.figure(figsize=(21, 9), dpi=500)
plt.bar(
    cobalt["index"],
    cobalt["daten"],
    linewidth=2,
    width=1.1,
    label=r"$^{60}\mathrm{Co}$",
    color="royalblue",
)
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="orange", label="Peaks")
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(cobalt["daten"].min(), cobalt["daten"].max(), 10))

plt.ylim(cobalt["daten"].min() - 1)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Cobalt-Peaks.pdf")
plt.clf()

# Kleinformatigere Abbildungen ab hier
matplotlib.rcParams.update({"font.size": 8})

grenzen = pd.DataFrame(data={"L": [14, 14], "R": [9, 9]})
zorder1 = [1, 2]
zorder2 = [1, 2]

for i in range(len(peaks)):
    fitmaker_2000(
        cobalt,
        peaks,
        grenzen["L"][i],
        grenzen["R"][i],
        i,
        scaled_gauss_cdf_b,
        "cobalt",
        zorder1=zorder1[i],
        zorder2=zorder2[i],
    )
    # Frag nicht
    # peaks.at[i, "N"] = peaks.at[i, "s"]
    # N = peaks.at[
    #     i, "N"
    # ]  # nur benötigt weil fstrings wenig flexibel sind; ein "" innerhalb des {} macht schon alles kaputt
    # print(f"N_{i+1} = {N:.5f}")

plt.close()
