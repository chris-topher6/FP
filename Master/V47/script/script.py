import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from uncertainties import ufloat

# Pandas dazu zwingen, alle Spalten anzuzeigen
pd.set_option("display.max_columns", None)

# Konstanten definieren
MOLARE_MASSE_CU = 0.06355  # M [kg/mol]
PROBENMASSE = 0.342  # m [kg]
PROBENDICHTE = 8.96  # rho [g/cm^3]
KOMPRESSIONSMODUL = 137.8  # kappa [GPa]

messung = pd.read_csv("./data/Messung.csv", skiprows=1, header=None)
messung.columns = ["t", "R_G", "R_P", "I", "U"]

# Umrechnung der Widerstände in Ohm
messung["R_G"] = messung["R_G"] * 100
messung["R_P"] = messung["R_P"] * 100
# t [min], R_G [Ohm], R_P [Ohm], I [mA], U [V]

# Berücksichtige limitierte Ablesegenauigkeit
messung["I_err"] = messung["I"].apply(lambda x: ufloat(x, 0.5))
messung["U_err"] = messung["U"].apply(lambda x: ufloat(x, 0.5))
messung["t_err"] = messung["t"].apply(lambda x: ufloat(x, 0.05))


def temp_berechnung(R):
    """
    Funktion, die aus einem gegebenen Widerstand die zugehörige Temperatur der Probe berechnet.
    Args:
    R: Widerstand in Ohm
    Returns:
    T: Temperatur in *C
    """
    T = 0.00134 * R**2 + 2.296 * R - 243.02
    return T


# Umrechnung in Kelvin
messung["T_G"] = temp_berechnung(messung["R_G"]) + 273.15
messung["T_P"] = temp_berechnung(messung["R_P"]) + 273.15
messung["T"] = (messung["T_G"] + messung["T_P"]) / 2
# t [min], R_G [Ohm], R_P [Ohm], I [mA], U [V], T_G [K], T_P [K], T[K]

# Für die Formel für C_p brauchen wir die Zeitabstände Delta t...
messung["Deltat"] = messung["t_err"].diff()
messung["Deltat"].fillna(1, inplace=True)  # ACHTUNG die 1 ist nur als Platzhalter da
# ...sowie die Temperaturabstände Delta T:
messung["DeltaT"] = messung["T"].diff()
messung["DeltaT"].fillna(1, inplace=True)  # ACHTUNG die 1 ist nur als Platzhalter da
# t [min], R_G [Ohm], R_P [Ohm], I [mA], U [V], T_G [K], T_P [K], T[K] Deltat [min], DeltaT [K]

# Damit am Ende [C_p] = [J/(mol*K)] rauskommt: I in [A], Deltat in [s] umrechnen:
messung["Deltat"] = messung["Deltat"] * 60
messung["I_err"] = messung["I_err"] / 1000
# t [min], R_G [Ohm], R_P [Ohm], I [A], U [V], T_G [K], T_P [K], T[K] Deltat [s], DeltaT [K]


def Cp_berechnen(U, I, Deltat, DeltaT, M, m):
    """
    Funktion die die Wärmekapazität bei konstantem Druck aus den folgenden Parametern berechnet:
    Args:
    U: Spannung [V]
    I: Strom [A]
    Deltat: Zeitabstand, [s]
    DeltaT: Temperaturänderung, [K]
    M: molare Masse der Probe, [kg/mol]
    m: Masse der Probe, [kg]
    Returns:
    C_p: Wärmekapazität bei konstantem Druck [J / (mol*K)]
    """
    C_p = (U * I * Deltat * M) / (DeltaT * m)
    return C_p


messung["C_p"] = Cp_berechnen(
    messung["U_err"],
    messung["I_err"],
    messung["Deltat"],
    messung["DeltaT"],
    MOLARE_MASSE_CU,
    PROBENMASSE,
)

# Bestimme Alpha durch Interpolation
werte = pd.read_csv("./data/Extrapolation_Alpha.csv", skiprows=1, header=None)
werte.columns = ["T", "alpha"]

# Interpoliere Daten mit kubischen Splines
# (die Sachen aus Numerik gibts tatsächlich in der Realität????)
cspl = CubicSpline(werte["T"], werte["alpha"])
xachse = np.linspace(werte["T"].min(), werte["T"].max(), 100)

plt.figure(figsize=(9, 5))
plt.plot(
    xachse,
    cspl(xachse),
    label="Cubic Spline",
    color="orange",
    linewidth=2.2,
    zorder=1,
)
plt.plot(
    werte["T"],
    werte["alpha"],
    "x",
    label="data",
    color="royalblue",
)
plt.legend()
plt.xlabel(r"$\mathrm{Temperature}/ \, \mathrm{K}$")
plt.ylabel(r"$\alpha / \, 10^{-6} / \mathrm{deg}$")
plt.tight_layout()
plt.savefig("./build/Alpha-Extrapolation.pdf")
plt.clf()


# Interpolierte Alpha Werte für jede Temperatur dem Dataframe hinzufügen
messung["alpha"] = cspl(messung["T"])


def CV_berechnen(C_p, alpha, kappa, V0, T):
    """
    Funktion zur Berechnung der Wärmekapazität bei konstantem Volumen aus der Wärmekapazität bei konstantem Druck.
    Args:
    C_p: isobare Wärmekapazität, [J/(mol * K)]
    alpha: Koeffizient der thermischen Ausdehnung
    kappa: Kompressionsmodul des Kupfers
    V0: Volumen??
    T: Temperatur
    """
    C_V = C_p - 9 * alpha**2 * kappa * V0 * T
    return C_V


# Frag mich nicht warum
V0 = 7.11 * 10 ** (-6)

messung["C_V"] = CV_berechnen(
    messung["C_p"], messung["alpha"], KOMPRESSIONSMODUL, V0, messung["T"]
)

# Plot aller in CV_berechnen verwendeten (von T abh.) Größen
plt.figure(figsize=(9, 5))
plt.plot(
    messung["T"],
    [v.n for v in messung["C_V"]],  # uncertainties mag pandas nicht
    label=r"$C_V$",
    color="royalblue",
    marker="x",
    linestyle="None",
)
plt.plot(
    messung["T"],
    [v.n for v in messung["C_p"]],
    label=r"$C_p$",
    color="orange",
    marker="x",
    linestyle="None",
)
plt.legend()
plt.xlabel(r"$\mathrm{Temperature}/ \, \mathrm{K}$")
plt.ylabel(r"$C/ \, \frac{\mathrm{J}}{\mathrm{kg} \cdot \mathrm{K}}$")
plt.tight_layout()
plt.savefig("./build/Params_CV_berechnen.pdf")
plt.clf()

# Theta interpolieren
werte_theta_arsch = pd.read_csv(
    "./data/Theta_aber_scheisse.csv", skiprows=1, header=None
)
# Ich hasse dieses Tabellenformat
index = np.arange(0, 16.0, 0.1)
werte_theta = pd.DataFrame(columns=["Theta", "C_V"])
werte_theta["Theta"] = index
werte_theta["C_V"] = pd.Series(werte_theta_arsch.values.flatten())

# CubicSpline braucht monoton steigende x-Werte, C_V fällt aber; daher umdrehen
werte_theta = werte_theta.sort_values(by="C_V")

# C_V von J/(mol K) in J/(g K) umrechnen; hierfür muss mit der molaren Masse multipliziert werden
# werte_theta["C_V"] = werte_theta["C_V"] * MOLARE_MASSE_CU
# # in J/(kg K)
# werte_theta["C_V"] = werte_theta["C_V"] * 1000

# Interpolieren
cspl2 = CubicSpline(werte_theta["C_V"], werte_theta["Theta"])
xachse2 = np.linspace(werte_theta["C_V"].min(), werte_theta["C_V"].max(), 100)

plt.figure(figsize=(9, 5))
plt.plot(
    xachse2,
    cspl2(xachse2),
    label="Cubic Spline",
    color="orange",
    zorder=2,
)
plt.plot(
    werte_theta["C_V"],
    werte_theta["Theta"],
    "x",
    label="data",
    color="royalblue",
    markersize=5,
    zorder=1,
)
plt.legend()
plt.xlabel(r"$C_V/ \,\frac{\mathrm{J}}{\mathrm{g} \cdot \mathrm{K}}$")
plt.ylabel(r"$\frac{\theta_{D}}{T}$")
plt.tight_layout()
plt.savefig("./build/Theta-Extrapolation.pdf")
plt.clf()

messung["Theta/T"] = cspl2([v.n for v in messung["C_V"]])
messung["Theta"] = messung["Theta/T"] * messung["T"]

# Schön exportieren
messung.to_csv("./build/Ergebnisse.csv")

print(messung)
print(messung.describe())
plt.close()
