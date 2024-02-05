import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from uncertainties import ufloat

# Konstanten definieren
MOLARE_MASSE_CU = 63.546  # M [g/mol]
PROBENMASSE = 342  # m [g]
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


def Cp_berechnen(U, I, Deltat, DeltaT, M, m):
    """
    Funktion die die Wärmekapazität bei konstantem Druck aus den folgenden Parametern berechnet:
    Args:
    U: Spannung
    I: Strom
    Deltat: Zeitabstand
    DeltaT: Temperaturänderung
    M: molare Masse der Probe
    m: Masse der Probe
    Returns:
    C_p: Wärmekapazität bei konstantem Druck [J * mol /]
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

plt.figure()
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
plt.xlabel(r"$\mathrm{Temperature}/K$")
plt.ylabel(r"$\alpha \cdot 10^{-6} / \mathrm{deg}$")
plt.tight_layout()
plt.savefig("./build/Alpha-Fit.pdf")
plt.clf()

# Interpolierte Alpha Werte für jede Temperatur dem Dataframe hinzufügen
messung["alpha"] = cspl(messung["T"])


def CV_berechnen(C_p, alpha, kappa, V0, T):
    """
    Funktion zur Berechnung der Wärmekapazität bei konstantem Volumen aus der Wärmekapazität bei konstantem Druck.
    Args:
    C_p: isobare Wärmekapazität
    alpha: Koeffizient der thermischen Ausdehnung
    kappa: Kompressionsmodul des Kupfers
    V0:
    T: Temperatur
    """
    C_V = C_p - 9 * alpha**2 * kappa * V0 * T
    return C_V


print(messung)
print(messung.describe())
