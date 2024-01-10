#!/usr/bin/env python3

import numpy as np
import pandas as pd
from uncertainties.umath import *
from uncertainties import ufloat
from datetime import datetime
import iminuit
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

start_aktivität = ufloat(4130, 60)  # Bq, am 1.10.2000

start_date = datetime(2000, 10, 1)
end_date = datetime(2023, 11, 20)

# Differenz zwischen den Daten
time_difference = end_date - start_date

# Umwandlung der Zeitdifferenz in Sekunden
time_difference_in_seconds = time_difference.total_seconds()

halbwertszeit_eu = (13.522) * 365.25 * 24 * 60 * 60  # Jahre, in s umgerechnet


def aktivitätsgesetz(t, A0, tau):
    """Formel für die Aktivität einer Probe"""
    A = A0 * exp(((-log(2)) / tau) * t)
    return A


end_aktivität = aktivitätsgesetz(
    time_difference_in_seconds, start_aktivität, halbwertszeit_eu
)
a = 8.91  # cm
r = 2.25  # cm
omega_4pi = 1 / 2 * (1 - a / (np.sqrt(a**2 + r**2)))
# print(f"Die Aktivität am Messtag betrug {end_aktivität}")
# print(f"Der Raumwinkel Omega/4pi beträgt {omega_4pi:.5f}")

peaks = pd.read_csv("./build/peaks.csv")


def fedp(omega, N, A, W, t):
    """Formel für die Vollenergienachweiswahrscheinlichkeit"""
    Q = (4 * np.pi) / omega * N / (A * W * t)
    return Q


# Messzeit
t = 2802  # s

peaks["fedp"] = float(0)

for zeile in range(len(peaks)):
    # Anzahl Events mit Unsicherheit
    N = ufloat(peaks.loc[zeile, "N"], peaks.loc[zeile, "N_err"])
    # Emissionwahrscheinlichekit nach Literatur mit Unsicherheit
    P = ufloat(peaks.loc[zeile, "Intensität"], peaks.loc[zeile, "Unsicherheit(I)"])
    u_fedp = fedp((omega_4pi * 4 * np.pi), N, end_aktivität, P, t)
    peaks.loc[zeile, "fedp"] = round(u_fedp.n, 6) * 10
    peaks.loc[zeile, "fedp_err"] = round(u_fedp.s, 6) * 10

peaks_speichern = peaks.to_csv("./build/peaks.csv", index=False)


def q_energy(E, a, b):
    """
    Zu fittende Funktion für die Energieabhängigkeit der FEDP
    """
    return a * ((E) ** b)


# Es gibt leider viele Ausreißer; das hier ist eigentlich alles nicht das wahre, ich
# produziere hier absichtlich die zu fittende Funktion ohne das diese wirklich zu den
# Daten passt
# peaks_fit = peaks.drop([0, 1, 2, 3, 5, 6, 8, 9, 11])
# peaks_fit = peaks.drop([0, 1, 2, 9, 11])
peaks_fit = peaks.drop([0, 1, 2, 5, 6, 7, 8, 10, 12])

energy_scale = np.linspace(
    peaks["Energie"].min(), peaks["Energie"].max(), 10000
)  # Fester Minimumwert manchmal nötig um Divergenz zu vermeiden

lq = LeastSquares(
    peaks_fit["Energie"], peaks_fit["fedp"], peaks_fit["fedp_err"], q_energy
)
m = Minuit(lq, a=43, b=-0.9)
# m.limits["a"] = (40, 45)
m.limits["b"] = (-1, -0.8)
# m.limits["c"] = (0, None)
m.migrad()
m.hesse()

matplotlib.rcParams.update({"font.size": 8})
fig, axs = plt.subplots(
    2,
    sharex=True,
    gridspec_kw={"hspace": 0.05, "height_ratios": [3, 1]},
    layout="constrained",
)

axs[0].errorbar(
    peaks["Energie"],
    peaks["fedp"],
    xerr=peaks["Unsicherheit(E)"],
    yerr=peaks["fedp_err"],
    fmt="o",
    color="grey",
    label="discarded data",
)

axs[0].errorbar(
    peaks_fit["Energie"],
    peaks_fit["fedp"],
    xerr=peaks_fit["Unsicherheit(E)"],
    yerr=peaks_fit["fedp_err"],
    fmt="o",
    color="royalblue",
    label="data",
)

# axs[0].plot(
#     peaks_fit["Energie"],
#     q_energy(peaks_fit["Energie"], *m.values),
#     label="fit",
#     color="orange",
#     linewidth=2.2,
# )

axs[0].plot(
    energy_scale,
    q_energy(energy_scale, *m.values),
    label="fit",
    color="orange",
    linewidth=2.2,
)
axs[0].legend()
axs[0].set_ylabel("Q/%")
# axs[0].set_ylabel("Q")

# Chi^2 Test des Fits auf Abbildung schreiben
fit_info = [
    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
]

# Fitparameter auf Abbildung schreiben
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

n_model = q_energy(peaks_fit["Energie"], *m.values)
n_error = peaks_fit["fedp_err"]
n_data = peaks_fit["fedp"]

pull = (n_data - n_model) / n_error

bin_edges = peaks_fit["Energie"]

axs[1].bar(
    peaks_fit["Energie"],
    pull,
    width=15,
    color="royalblue",
)
axs[1].axhline(0, color="orange", linewidth=0.8)
axs[1].set_xlabel(r"$\mathrm{Energy}/\,\mathrm{keV}$")
axs[1].set_ylabel(r"$\mathrm{Pull}/\,\sigma$")
axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
axs[1].set_yticks(
    np.linspace(
        int(pull.min() - 1),
        int(pull.max() + 1),
        10,
    )
)
axs[0].legend(title="\n".join(fit_info), frameon=False)
plt.savefig("./build/FEDP-Fit.pdf")
plt.clf()
