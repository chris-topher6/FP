#!/usr/bin/env python3

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
import scipy.constants as c

# Messwerte importieren
df0 = pd.read_csv("../data/magnetfeld.csv")
# Probe 1: GaAs, n-dotiert, N=1.2*10^18 cm^-3, d=1.36mm
df1 = pd.read_csv("../data/probe1.csv")
# Probe 2: GaAs, n-dotiert, N=2.8*10^18 cm^-3, d=1.296mm
df2 = pd.read_csv("../data/probe2.csv")
# Probe 3: GaAs, hochrein, d=5.1mm
df3 = pd.read_csv("../data/probe3.csv")

# 1. Magnetfeldstärkenmaximum bestimmen

plt.figure()
plt.plot(
    df0["Position Hallsonde [mm]"],
    df0[" Feldstärke [mT]"],
    marker="x",
    ls="",
    color="red",
    label="Messdaten",
)
plt.axhline(y=434, ls="--", label="Maximum", color="black")
plt.ylabel(f"$B$ [mT]")
plt.xlabel("$d$ [mm]")
plt.title("Magnetische Flussdichte")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../build/B_Feld.pdf")
plt.close()

# 2.Graphische Darstellung der Messergebnisse
# Mittelung über beide Polarisierungen des Magnetfeldes
df1["Theta"] = np.abs(
    0.5
    * (
        df1[" Magnetfeld Max. normale Polung [grad]"]
        - df1[" Magnetfeld Max. umgepolt [grad]"]
    )
)
df2["Theta"] = np.abs(
    0.5
    * (
        df2[" Magnetfeld Max. normale Polung [grad]"]
        - df2[" Magnetfeld Max. umgepolt [grad]"]
    )
)
df3["Theta"] = np.abs(
    0.5
    * (
        df3[" Magnetfeld Max. normale Polung [grad]"]
        - df3[" Magnetfeld Max. umgepolt [grad]"]
    )
)
# Normierung der Winkel auf die Probendicken, Umwandlung Grad -> Radiant
df1["Theta"] = ((df1["Theta"] / 1.36) / 180) * np.pi  # d = 1.36mm
df2["Theta"] = ((df2["Theta"] / 1.296) / 180) * np.pi  # d = 1.296mm
df3["Theta"] = ((df3["Theta"] / 5.1) / 180) * np.pi  # d = 5.1mm

# Unnötige columns rausschmeißen
df1 = df1.drop(
    columns=[
        " Magnetfeld Max. normale Polung [grad]",
        " Magnetfeld Max. umgepolt [grad]",
    ]
)
df2 = df2.drop(
    columns=[
        " Magnetfeld Max. normale Polung [grad]",
        " Magnetfeld Max. umgepolt [grad]",
    ]
)
df3 = df3.drop(
    columns=[
        " Magnetfeld Max. normale Polung [grad]",
        " Magnetfeld Max. umgepolt [grad]",
    ]
)

# Drehwinkel der freien Elektronen bestimmen
df1["Theta frei"] = np.abs(df1["Theta"] - df3["Theta"])
df2["Theta frei"] = np.abs(df2["Theta"] - df3["Theta"])

# plotten
plt.figure()
plt.plot(
    df1["Filter [mikro m]"] ** 2,
    np.abs(df1["Theta"]),
    marker="x",
    ls="",
    label=r"Messdaten N = $1.2 \cdot 10^{18} cm^{-3}$",
    color="#1f77b4",
)
plt.plot(
    df2["Filter [mikro m]"] ** 2,
    np.abs(df2["Theta"]),
    marker="x",
    ls="",
    label=r"Messdaten N = $2.8 \cdot 10^{18} cm^{-3}$",
    color="#ff7f0e",
)
plt.plot(
    df3["Filter [mikro m]"] ** 2,
    np.abs(df3["Theta"]),
    marker="x",
    ls="",
    label="Messdaten GaAs hochrein",
    color="#2ca02c",
)
plt.ylabel(r"$\theta_{norm}$ [rad]")
plt.xlabel(r"$\lambda^2$  [$\mu m^2$]")
plt.title("Normierte Drehwinkel")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../build/Drehwinkel.pdf")
plt.close()

# 3. Graphische Darstellung des Drehwinkels der freien Elektronen


def f(x, m, b):
    return m * x + b


# Fit an den Daten der Probe 1
params1, cov1 = curve_fit(f, df1["Filter [mikro m]"] ** 2, df1["Theta frei"])
errors1 = np.sqrt(np.diag(cov1))
params1_err = unp.uarray(params1, errors1)


plt.figure()
plt.plot(
    (df1["Filter [mikro m]"] ** 2),
    df1["Theta frei"],
    marker="x",
    ls="",
    color="red",
    label=r"Drehwinkel $\theta_{frei}$",
)
plt.plot(
    np.linspace(1, 7, 100),
    f(np.linspace(1, 7, 100), params1[0], params1[1]),
    ls="-",
    label=r"Fit",
    color="#1f77b4",
)
plt.ylabel(r"$\theta_{frei}$ [rad]")
plt.xlabel(r"$\lambda^2$  [$\mu m^2$]")
plt.title("Drehwinkel der freien Elektronen für Probe 1")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../build/Drehwinkel_frei_Probe1.pdf")
plt.close()

# Fit an den Daten der Probe 2
params2, cov2 = curve_fit(f, df2["Filter [mikro m]"] ** 2, df2["Theta frei"])
errors2 = np.sqrt(np.diag(cov2))
params2_err = unp.uarray(params2, errors2)

plt.figure()
plt.plot(
    (df2["Filter [mikro m]"] ** 2),
    df2["Theta frei"],
    marker="x",
    ls="",
    color="red",
    label=r"Drehwinkel $\theta_{frei}$",
)

plt.plot(
    np.linspace(1, 7, 100),
    f(np.linspace(1, 7, 100), params2[0], params2[1]),
    ls="-",
    label=r"Fit",
    color="#1f77b4",
)

plt.ylabel(r"$\theta_{frei}$ [rad]")
plt.xlabel(r"$\lambda^2$  [$\mu m^2$]")
plt.title("Drehwinkel der freien Elektronen für Probe 2")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../build/Drehwinkel_frei_Probe2.pdf")
plt.close()

# Ausreißer herausnehmen, um den Fit zu verbessern
df1_v2 = df1.drop([7], axis=0)
df1_antiv2 = df1.drop([0, 1, 2, 3, 4, 5, 6, 8], axis=0)

# Fit an den Daten der Probe 1
params1v2, cov1v2 = curve_fit(f, df1_v2["Filter [mikro m]"] ** 2, df1_v2["Theta frei"])
errors1v2 = np.sqrt(np.diag(cov1v2))
params1v2_err = unp.uarray(params1v2, errors1v2)


plt.figure()
plt.plot(
    (df1_v2["Filter [mikro m]"] ** 2),
    df1_v2["Theta frei"],
    marker="x",
    ls="",
    color="red",
    label=r"Drehwinkel $\theta_{frei}$",
)
plt.plot(
    (df1_antiv2["Filter [mikro m]"] ** 2),
    df1_antiv2["Theta frei"],
    marker="x",
    ls="",
    color="black",
    label="Ausreißer",
)
plt.plot(
    np.linspace(1, 7, 100),
    f(np.linspace(1, 7, 100), params1v2[0], params1v2[1]),
    ls="-",
    label=r"Fit",
    color="#1f77b4",
)
plt.ylabel(r"$\theta_{frei}$ [rad]")
plt.xlabel(r"$\lambda^2$  [$\mu m^2$]")
plt.title("Drehwinkel der freien Elektronen für Probe 1")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("../build/Drehwinkel_frei_Probe1v2.pdf")
plt.close()

# Berechnung der effektiven Masse
# Konstanten definieren
B = 434 * 10 ** (-3)  # von mT in T umrechnen
n = 3.57  # aus Altprotokoll, Quelle raussuchen und ersetzen
N1 = 1.2 * 10 ** (24)  # von  cm^-3 in m^-3 umrechnen
N2 = 2.8 * 10 ** (24)  # von cm^-3 in m^-3 umrechnen
params1v2_err[0] *= 10 ** (12)  # von radian/micro m^3 in radian/m^3 umrechnen
params2_err[0] *= 10 ** (12)  # von radian/micro m^3 in radian/m^3 umrechnen
print("Es ergeben sich die Proportionalitätsfaktoren:")
print("Probe 1: ")
print("m = ", params1v2_err[0], "radian/m^3")
print("b = ", params1_err[1])
print("Probe 2: ")
print("m = ", params2_err[0], "radian/m^3")
print("b = ", params2_err[1])
# Berechnung
print("Es ergeben sich die effektiven Massen:")
m1 = unp.sqrt(
    (c.e**3 * B)
    / (8 * np.pi**2 * c.epsilon_0 * c.c**3 * n)
    * (N1 / params1v2_err[0])
)  # in kg
m2 = unp.sqrt(
    (c.e**3 * B)
    / (8 * np.pi**2 * c.epsilon_0 * c.c**3 * n)
    * (N2 / params2_err[0])
)  # in kg
print("m1: ", m1, " kg")
print("m2: ", m2, " kg")
print("In Elektronenmassen ausgedrückt: ")
print("m1: ", m1 / c.m_e, "* m_e")
print("m2: ", m2 / c.m_e, "* m_e")
