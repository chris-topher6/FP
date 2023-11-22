#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


# Einlesen der Kalibrationsmessung
SKIP_ANFANG = 12
SKIP_ENDE = 14

europium = pd.read_csv("./data/Eu1.Spe", skiprows=SKIP_ANFANG, header=None)
europium = europium.iloc[:-SKIP_ENDE]  # Entferne den Rotz am Ende
europium.columns = ["data"]
europium["index"] = europium.index

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
europium["data"] = pd.to_numeric(europium["data"], errors="coerce")

# Plot der Kalibrationsmessung
plt.figure(figsize=(21, 9))

plt.bar(europium["index"], europium["data"], linewidth=2, width=1.1)

plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(europium["data"].min(), europium["data"].max(), 10))

plt.ylim(europium["data"].min() - 30)

plt.xlabel(r"Kanäle")
plt.ylabel(r"Signale")

plt.grid(True, linewidth=0.1)

plt.savefig("./build/Europium-Test.pdf")
