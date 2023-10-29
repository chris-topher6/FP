#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Verwende LaTeX für Textrendering
rcParams["text.usetex"] = True

messung1 = pd.read_csv("./data/Distanz-Intensität-Messung1.txt", sep="\s+", header=1)
messung2 = pd.read_csv("./data/Distanz-Intensität-Messung2.txt", sep="\s+", header=1)

# Längenmessungen haben auf jeder Seite 4cm zuviel
messung1["d"] -= 8
messung2["d"] -= 8

plt.plot(
    messung1["d"],
    messung1["I"],
    marker="x",
    markeredgecolor="orangered",
    color="lightsalmon",
    linewidth="0.8",
    label=(r"$\mathrm{Flat}/1400\mathrm{mm} - \mathrm{Flat}/1400\mathrm{mm}$"),
)
plt.plot(
    messung2["d"],
    messung2["I"],
    marker="x",
    markeredgecolor="royalblue",
    color="lightsteelblue",
    linewidth="0.8",
    label=(r"$\mathrm{Flat}/\mathrm{Flat} - \mathrm{Flat}/1400\mathrm{mm}$"),
)
plt.axvline(
    x=280,
    color="lightsalmon",
    linewidth="0.8",
    linestyle="--",
    label=r"$\mathrm{Theoretical\ maximum}$",
)
plt.axvline(
    x=140,
    color="lightsteelblue",
    linestyle="--",
    linewidth="0.8",
    label=r"$\mathrm{Theoretical\ maximum}$",
)
plt.xlabel(r"$\mathrm{Distance/cm}$")
plt.ylabel(r"$\mathrm{Intensity/mW}$")
plt.legend()
# plt.grid(True)
plt.savefig("./build/Distanz-Intensität.pdf")
plt.clf()
