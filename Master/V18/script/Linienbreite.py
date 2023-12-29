#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares

matplotlib.rcParams.update({"font.size": 18})

SKIP_ANFANG = 12
SKIP_ENDE = 14

# Einlesen der Kalibrations- und Untergrundmessung
europium_raw = pd.read_csv("./data/Eu1.Spe", skiprows=SKIP_ANFANG, header=None)
europium_raw.columns = ["daten"]
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]

# Rotz am Ende entfernen
europium_raw = europium_raw.iloc[:-SKIP_ENDE]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
europium_raw["daten"] = pd.to_numeric(europium_raw["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index hinzufügen
europium_raw["index"] = europium_raw.index
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
plt.yticks(np.linspace(europium_raw["daten"].min(), europium_raw["daten"].max(), 10))

plt.ylim(europium_raw["daten"].min() - 30)

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
    europium_raw["index"],
    europium_raw["daten"],
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
plt.yticks(np.linspace(europium_raw["daten"].min(), europium_raw["daten"].max(), 10))

plt.ylim(europium_raw["daten"].min() - 30)

plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Europium-Untergrund.pdf")
plt.clf()

# Untergrund entfernen
europium_raw["daten"] = europium_raw["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
europium_raw["daten"] = europium_raw["daten"].clip(lower=0)

# Vorher bestimmte Peaks einlesen
peaks = pd.read_csv("./build/peaks.csv")

print(europium_raw)
print(peaks)
