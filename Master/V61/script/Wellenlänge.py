#!/usr/bin/env python3

import numpy as np
import pandas as pd

# Daten einlesen
a1 = pd.read_csv("./data/Wellenlänge1.txt", sep="\s+", header=1)
a2 = pd.read_csv("./data/Wellenlänge2.txt", sep="\s+", header=1)
a3 = pd.read_csv("./data/Wellenlänge3.txt", sep="\s+", header=1)
a4 = pd.read_csv("./data/Wellenlänge4.txt", sep="\s+", header=1)

asigma = 0.2  # in cm

a1["ak"] = a1["ak"] / 2
a1.rename(columns={"2k": "k"}, inplace=True)
a2["ak"] = a2["ak"] / 2
a2.rename(columns={"2k": "k"}, inplace=True)
a3["ak"] = a3["ak"] / 2
a3.rename(columns={"2k": "k"}, inplace=True)
a4["ak"] = a4["ak"] / 2
a4.rename(columns={"2k": "k"}, inplace=True)


def lambdaa(d, k, ak, e):
    lambdaa = (d * ak) / (k * np.sqrt(e**2 + ak**2))
    return lambdaa


a1["lambdaa"] = lambdaa(800, a1["k"], a1["ak"], 70)  # 80mm^-1 = 800cm^-1
a2["lambdaa"] = lambdaa(1000, a2["k"], a2["ak"], 70)
a3["lambdaa"] = lambdaa(6000, a3["k"], a3["ak"], 29.4)
a4["lambdaa"] = lambdaa(12000, a4["k"], a4["ak"], 29.4)

a1mean = a1["lambdaa"].mean() * 10  # in nm umrechnen
a2mean = a2["lambdaa"].mean() * 10  # in nm umrechnen
a3mean = a3["lambdaa"].mean() * 10  # in nm umrechnen
a4mean = a4["lambdaa"].mean() * 10  # in nm umrechnen
amean = (a1mean + a2mean + a3mean + a4mean) / 4
a12mean = (a1mean + a2mean) / 2

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

    file.write("Wellenlänge aus den ersten beiden Messreihen gemittelt:\n")
    file.write(f"lambda ={a12mean:.4f} nm \n")
