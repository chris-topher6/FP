#!/usr/bin/env python3

import numpy as np
import pandas as pd
import uncertainties
from uncertainties import unumpy as unp, ufloat

# Daten einlesen
a1 = pd.read_csv("./data/Wellenlänge1.txt", sep="\s+", header=1)
a2 = pd.read_csv("./data/Wellenlänge2.txt", sep="\s+", header=1)
a3 = pd.read_csv("./data/Wellenlänge3.txt", sep="\s+", header=1)
a4 = pd.read_csv("./data/Wellenlänge4.txt", sep="\s+", header=1)

asigma = 0.2  # in cm

# Geschätzter Messfehler
a1["ak"] = unp.uarray(a1["ak"], asigma)
a2["ak"] = unp.uarray(a2["ak"], asigma)
a3["ak"] = unp.uarray(a3["ak"], asigma)
a4["ak"] = unp.uarray(a4["ak"], asigma)

a1["ak"] = a1["ak"] / 2
a1.rename(columns={"2k": "k"}, inplace=True)
a2["ak"] = a2["ak"] / 2
a2.rename(columns={"2k": "k"}, inplace=True)
a3["ak"] = a3["ak"] / 2
a3.rename(columns={"2k": "k"}, inplace=True)
a4["ak"] = a4["ak"] / 2
a4.rename(columns={"2k": "k"}, inplace=True)


def lambdaa(d, k, ak, e):
    lambdaa = (d * ak) / (k * unp.sqrt(e**2 + ak**2))
    return lambdaa


def wavg(a,b):
    """Einfacher Weg, einen gewichteten Mittelwert zweier fehlerbehafteter Größen zu berechnen"""
    wavg = ufloat((a.n/a.s**2 + b.n/b.s**2)/(1/a.s**2 + 1/b.s**2),np.sqrt(2/(1/a.s**2 + 1/b.s**2)))
    return wavg


a1["lambdaa"] = lambdaa(800, a1["k"], a1["ak"], 70) * 10  # in nm umrechnen;   80mm^-1 = 800cm^-1
a2["lambdaa"] = lambdaa(1000, a2["k"], a2["ak"], 70) *10
a3["lambdaa"] = lambdaa(6000, a3["k"], a3["ak"], 29.4) *10
a4["lambdaa"] = lambdaa(12000, a4["k"], a4["ak"], 29.4)*10

a1mean = wavg(wavg(wavg(a1["lambdaa"][0], a1["lambdaa"][1]), wavg(a1["lambdaa"][2], a1["lambdaa"][3])), wavg(a1["lambdaa"][4], a1["lambdaa"][5]))
a2mean = wavg(wavg(wavg(a2["lambdaa"][0], a2["lambdaa"][1]), wavg(a2["lambdaa"][2], a2["lambdaa"][3])), wavg(a2["lambdaa"][4], a2["lambdaa"][5]))
a3mean = wavg(a3["lambdaa"][0], a3["lambdaa"][1])
a4mean = a4["lambdaa"][0]
amean = wavg(wavg(a1mean, a2mean), wavg(a3mean, a4mean))
a12mean = wavg(a1mean, a2mean)

alit = 632.8 # nm

abweichung = (unp.sqrt((amean-alit)**2)/alit)*100 # in %

with open("./build/wavelengths.txt", "w") as file:
    file.write("Wellenlänge aus der Messreihe 1 (d=70cm, 800l/cm):\n")
    file.write(f"lambda ={a1mean:.4f} nm \n")

    file.write("Wellenlänge aus der Messreihe 2 (d=70cm, 1000l/cm):\n")
    file.write(f"lambda ={a2mean:.4f} nm \n")

    file.write("Wellenlänge aus der Messreihe 3 (d=29.4cm, 6000l/cm):\n")
    file.write(f"lambda ={a3mean:.4f} nm \n")

    file.write("Wellenlänge aus der Messreihe 4 (d=29.4cm, 12000l/cm):\n")
    file.write(f"lambda ={a4mean:.4f} nm \n")

    file.write("Wellenlänge aus allen vier Messreihen gemittelt:\n")
    file.write(f"lambda ={amean:.4f} nm \n")

    file.write(f"Damit weicht das Ergebnis von dem Literaturwert alit=632.8nm um {abweichung:.4f}% ab.\n")

    file.write("Wellenlänge aus den ersten beiden Messreihen gemittelt:\n")
    file.write(f"lambda ={a12mean:.4f} nm \n")
