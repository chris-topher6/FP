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
from Kalibrierung import linear
from Kalibrierung import linear_invers


matplotlib.rcParams.update({"font.size": 18})

# Das hier sind die Fitparameter aus der Kalibrierung
alpha = ufloat(0.207452, 0)
beta = ufloat(-1.622490, 0.000381)

# Einlesen der Caesium Messung
SKIP_ANFANG = 12
SKIP_ENDE = 14

caesium = pd.read_csv("./data/Cs1.Spe", skiprows=SKIP_ANFANG, header=None)
caesium = caesium.iloc[:-SKIP_ENDE]
caesium.columns = ["daten"]

# Einlesen der Untergrundmessung
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
caesium["daten"] = pd.to_numeric(caesium["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index ergänzen
caesium["index"] = caesium.index
untergrund["index"] = untergrund.index

# Untergrundmessung skalieren; Cs Messung dauerte 2865s; Untergrundmessung
# dauerte 76353s
untergrund["daten"] = untergrund["daten"] * (2865 / 76353)

# Untergrund entfernen
caesium["daten"] = caesium["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
caesium["daten"] = caesium["daten"].clip(lower=0)

# Peaks raussuchen um sie zu fitten
peaks_array, peaks_params = find_peaks(
    caesium["daten"], height=20, prominence=40, distance=10
)
peaks = pd.DataFrame(peaks_params)
peaks["peaks"] = peaks_array

# Raus mit dem Müll
peaks = peaks.drop([0])
peaks = peaks.reset_index(drop=True)

# Schonmal neue Spalten anlegen
peaks["N"] = float(0)
peaks["N_err"] = float(0)

plt.figure(figsize=(21, 9), dpi=500)
plt.bar(
    caesium["index"],
    caesium["daten"],
    linewidth=2,
    width=1.1,
    label=r"$^{137}\mathrm{Cs}$",
    color="royalblue",
)
plt.plot(peaks["peaks"], peaks["peak_heights"], "x", color="orange", label="Peaks")
plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(caesium["daten"].min(), caesium["daten"].max(), 10))

plt.ylim(caesium["daten"].min() - 30)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Caesium-Peaks.pdf")
plt.clf()


# Berechnung der theoretischen Erwartung für die Comptonkante
def Eemax(egamma):
    """
    Funktion für die Energie des Elektrons bei Compton-Streuung mit theta=180 grad.
    """
    return egamma / (1 + (510.99895) / (2 * egamma))


egamma = ufloat(661.3941, 0.0004)
ecompton = Eemax(egamma)

print(f"E_e für theta=180 = {ecompton} keV")

# Fürs Plotten wieder in Kanäle umrechnen
ecompton_K = linear_invers(ecompton, alpha, beta)

# Um das Compton-Kontinuum und die -Kante besser sehen zu können den Bums nochmal plotten
plt.figure(figsize=(21, 9), dpi=500)
caesium_short = caesium[0:3300]  # weiter rechts ist eh nix
plt.bar(
    caesium_short["index"],
    caesium_short["daten"],
    linewidth=2,
    width=1.1,
    label=r"$^{137}\mathrm{Cs}$",
    color="royalblue",
    zorder=2,
)
plt.plot(
    peaks.at[0, "peaks"],
    peaks.at[0, "peak_heights"],
    "x",
    color="royalblue",
    label="Backscatter Peak",
    zorder=3,
)
plt.plot(
    peaks.at[1, "peaks"],
    peaks.at[1, "peak_heights"],
    "x",
    color="cornflowerblue",
    label="Photoelectric Peak",
    zorder=4,
)
plt.xticks(np.linspace(0, 3300, 10))
plt.yticks(np.linspace(caesium_short["daten"].min(), caesium_short["daten"].max(), 10))

plt.ylim(caesium_short["daten"].min() - 30)
plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

# plt.grid(True, linewidth=0.1)
plt.tight_layout()

# Bereich für die eingesetzte Ansicht definieren
x1, x2 = 2200, 2500
y1, y2 = (
    caesium_short[(caesium_short["index"] >= x1) & (caesium_short["index"] <= x2)][
        "daten"
    ].min(),
    caesium_short[(caesium_short["index"] >= x1) & (caesium_short["index"] <= x2)][
        "daten"
    ].max(),
)
plt.axvline(
    x=ecompton_K.n,
    color="mediumpurple",
    linewidth=2.2,
    label=r"$E_{\mathrm{Compton}}$ (Theoretical)",
    zorder=5,
)
plt.axvline(
    x=linear_invers(474.7, alpha, beta).n,
    color="orange",
    linewidth=2.2,
    label=r"$E_{\mathrm{Compton}}$ (Estimated)",
    zorder=6,
)
plt.fill_betweenx(
    y=[caesium_short["daten"].min() - 30, caesium_short["daten"].max() + 30],
    x1=1350,
    x2=linear_invers(474.7, alpha, beta).n,
    color="orange",
    alpha=0.2,
    zorder=1,
)
plt.axvline(
    x=1350,
    color="orange",
    linewidth=2.2,
    linestyle="dashed",
    zorder=6,
)
plt.legend()
# Eingesetzte Ansicht erstellen
ax = plt.gca()
ax_inset = inset_axes(
    ax,
    width="50%",
    height="50%",
    loc=3,
    bbox_to_anchor=(0.5, 0.5, 0.4, 0.4),
    bbox_transform=ax.transAxes,
)  # Position und Größe der eingesetzten Ansicht anpassen

ax_inset.bar(
    caesium_short["index"],
    caesium_short["daten"],
    linewidth=2,
    width=1.1,
    color="royalblue",
    zorder=2,
)
ax_inset.axvline(
    x=ecompton_K.n,
    color="mediumpurple",
    linewidth=2.2,
    zorder=4,
)
ax_inset.axvline(
    x=linear_invers(474.7, alpha, beta).n,
    color="orange",
    linewidth=2.2,
    zorder=3,
)
ax_inset.fill_betweenx(
    y=[caesium_short["daten"].min() - 30, caesium_short["daten"].max() + 30],
    x1=1350,
    x2=linear_invers(474.7, alpha, beta).n,
    color="orange",
    alpha=0.2,
    zorder=1,
)
# Grenzen der eingesetzten Ansicht
ax_inset.set_xlim(x1, x2)
ax_inset.set_ylim(y1 - 10, y2 + 10)
# Markiere Bereich im Hauptplot
mark_inset(ax, ax_inset, loc1=1, loc2=3, fc="none", ec="0.5")

plt.savefig("./build/Caesium-Peaks-Short.pdf")
plt.clf()

print(
    f"Vom Compton-Effekt dominierter Bereich: von {linear(1350, alpha, beta):.3f} keV bis 474.7 keV"
)
summe_compton = caesium_short["daten"].sum()
print(f"Dieser Bereich enthält {summe_compton} Events")
print(f"Die 474.7 keV entsprechen einem Kanal von {linear_invers(474.7, alpha, beta)}")

matplotlib.rcParams.update({"font.size": 8})


def scaled_gauss_cdf_b(x, s, mu, sigma, b):
    """
    Eine skalierte kumulative Normalverteilungsfunktion,
    die an die Signale angepasst werden soll.

    Parameter:

    x: array-like, bin edges; muss len(n) + 1 sein (n: Anzahl bins)

    s: Skalierungsparameter der Funktion; ist auch der Integralwert

    mu: Erwartungswert der Normalverteilung

    sigma: Standardabweichung der Normalverteilung
    """
    return s * norm(mu, sigma).cdf(x) + b * x


def scaled_gauss_pdf_b(x, s, mu, sigma, b):
    """
    Eine skalierte Normalverteilungsfunktion,
    die an die Signale angepasst werden soll.

    Parameter:

    x: array-like, bin edges; muss len(n) + 1 sein (n: Anzahl bins)

    s: Skalierungsparameter der Funktion; ist auch der Integralwert

    mu: Erwartungswert der Normalverteilung

    sigma: Standardabweichung der Normalverteilung
    """
    return s * norm(mu, sigma).pdf(x) + b


grenzen = pd.DataFrame(data={"L": [150, 15], "R": [150, 15]})
zorder1 = [1, 2]
zorder2 = [2, 1]

# Das ist jetzt auch wieder arschcode aber wir wollen fertig werden
# Wenn jemand ganz viel Zeit hat wäre es viel sinnvoller die fitmaker_2000 Funktion so zu schreiben
# dass sie die Fitparameter einfach zurückgibt und die ganze Logik die das schreiben in ein Dataframe
# und speichern als csv übernimmt einfach aus der Funktion auslagert
# Das hier sind die Fitparameter die weiter unten bestimmt wurden
peaks["s"] = pd.NA
peaks["b"] = pd.NA
peaks["mu"] = pd.NA
peaks["sigma"] = pd.NA
peaks.at[0, "s"] = ufloat(1275.237, 70.842)
peaks.at[1, "s"] = ufloat(9282.567, 98.438)
peaks.at[0, "b"] = ufloat(20.0, 0.457)
peaks.at[1, "b"] = ufloat(15.0, 0.165)
peaks.at[0, "mu"] = ufloat(929.282, 1.989)
peaks.at[1, "mu"] = ufloat(3195.882, 0.050)
peaks.at[0, "sigma"] = ufloat(31.257, 2.129)
peaks.at[1, "sigma"] = ufloat(4.494, 0.038)

peaks["N"] = pd.NA
for i in range(len(peaks)):
    fitmaker_2000(
        caesium,
        peaks,
        grenzen["L"][i],
        grenzen["R"][i],
        i,
        scaled_gauss_cdf_b,
        zorder1=zorder1[i],
        zorder2=zorder2[i],
    )
    # Frag nicht
    peaks.at[i, "N"] = peaks.at[i, "s"]
    N = peaks.at[
        i, "N"
    ]  # nur benötigt weil fstrings wenig flexibel sind; ein "" innerhalb des {} macht schon alles kaputt
    print(f"N_{i+1} = {N:.5f}")

# Umrechnung der Kanäle in Energien wurde in Kalibrierung.py bestimmt
peaks["Energie"] = linear(peaks["peaks"], alpha, beta)

# Wurden durch Fit bestimmt, updaten wenn nötig;
# Sind die Parameter des Fits für Q(E)
a = ufloat(4.757, 0.347)
b = ufloat(-0.915, 0.013)
peaks["Q"] = q_energy(peaks["Energie"], a, b)

# Bestimme die x-lims der Halbwertsbreite numerisch, geht bestimmt noch besser
# Kleine Abkürzung macht alles lesbarer
gauss_max = (
    peaks.at[1, "s"].n * 1 / (np.sqrt(2 * np.pi) * peaks.at[1, "sigma"].n)
    + peaks.at[1, "b"].n
)


def eq1(x):
    """
    Funktion die den Schnittpunkt zwischen Halbwertsbreite und Fit als Nullstelle hat
    """
    y = 0.5 * gauss_max - scaled_gauss_pdf_b(
        x,
        peaks.at[1, "s"].n,
        peaks.at[1, "mu"].n,
        peaks.at[1, "sigma"].n,
        peaks.at[1, "b"].n,
    )
    return y


def eq2(x):
    """
    Funktion die den Schnittpunkt zwischen Zehntelwertsbreite und Fit als Nullstelle hat
    """
    y = 0.1 * gauss_max - scaled_gauss_pdf_b(
        x,
        peaks.at[1, "s"].n,
        peaks.at[1, "mu"].n,
        peaks.at[1, "sigma"].n,
        peaks.at[1, "b"].n,
    )
    return y


# Lösungen für die Halbwertsbreite
solution_hm1 = fsolve(eq1, 3191)
solution_hm2 = fsolve(eq1, 3201)
# Lösungen für die Zenhntelwertsbreite
solution_tm1 = fsolve(eq2, 3185)
solution_tm2 = fsolve(eq2, 3205)

# Das hier ist die Halbwertsbreite von einer Normalverteilung
FWHM = 2 * np.sqrt(2 * np.log(2)) * peaks.at[1, "sigma"]
# und das hier die Zehntelwertsbreite
FWTM = 2 * np.sqrt(2 * np.log(10)) * peaks.at[1, "sigma"]
Verhältnis = FWHM / FWTM
Verhältnis_keV = linear(Verhältnis, alpha, beta)
print(f"Verhältnis FWHM/FWTM = {Verhältnis:.4f}")
print(f"Verhältnis FWHM/FWTM = {Verhältnis_keV:.4f} keV")

# Umrechnung nochmal in keV
FWHM_keV = linear(FWHM, alpha, beta)
FWTM_keV = linear(FWTM, alpha, beta)
print(f"FWHM = {FWHM_keV:.4f} keV")
print(f"FWTM = {FWTM_keV:.4f} keV")

plt.figure(figsize=(21, 9))
maske = (caesium["index"] >= peaks["peaks"][1] - 15) & (
    caesium["index"] <= peaks["peaks"][1] + 15
)
daten_cut = caesium[maske]
cut_bin_edges = np.arange(daten_cut["index"].min(), daten_cut["index"].max() + 2)
bin_centers = (cut_bin_edges[:-1] + cut_bin_edges[1:]) / 2
matplotlib.rcParams.update({"font.size": 18})

plt.fill_betweenx(
    y=[daten_cut["daten"].min() - 30, daten_cut["daten"].max() + 30],
    x1=int(solution_hm1),
    x2=int(solution_hm2),
    color="lightsteelblue",
    alpha=0.6,
)
plt.fill_betweenx(
    y=[daten_cut["daten"].min() - 30, daten_cut["daten"].max() + 30],
    x1=int(solution_tm1),
    x2=int(solution_tm2),
    color="lightsteelblue",
    alpha=0.4,
)
plt.errorbar(
    bin_centers,
    daten_cut["daten"],
    yerr=np.sqrt(daten_cut["daten"]),
    fmt="o",
    color="royalblue",
    label=r"$^{137}\mathrm{Cs}$ Photoeffect Peak",
    barsabove=True,
)
plt.stairs(
    np.diff(
        scaled_gauss_cdf_b(
            cut_bin_edges,
            peaks.at[1, "s"].n,
            peaks.at[1, "mu"].n,
            peaks.at[1, "sigma"].n,
            peaks.at[1, "b"].n,
        )
    ),
    cut_bin_edges,
    label="fit",
    color="orange",
    linewidth=3.5,
)
# Das hier ist die Halbwertsbreite
plt.hlines(
    y=0.5 * peaks.at[1, "s"].n * 1 / (np.sqrt(2 * np.pi) * peaks.at[1, "sigma"].n)
    + peaks.at[1, "b"].n,
    xmin=int(solution_hm1),
    xmax=int(solution_hm2),
    color="royalblue",
    label=f"FWHM with Δx = {FWHM:.4f}",
    linewidth=3.5,
)
plt.axvline(x=int(solution_hm1), color="royalblue", linewidth=3.0, linestyle="dashed")
plt.axvline(x=int(solution_hm2), color="royalblue", linewidth=3.0, linestyle="dashed")
# Das hier die Zehntelwertsbreite
plt.hlines(
    y=0.1 * peaks.at[1, "s"].n * 1 / (np.sqrt(2 * np.pi) * peaks.at[1, "sigma"].n)
    + peaks.at[1, "b"].n,
    xmin=int(solution_tm1),
    xmax=int(solution_tm2),
    color="cornflowerblue",
    label=f"FWTM with Δx = {FWTM:.4f}",
    linewidth=3.5,
)
plt.axvline(
    x=int(solution_tm1), color="cornflowerblue", linewidth=3.0, linestyle="dashed"
)
plt.axvline(
    x=int(solution_tm2), color="cornflowerblue", linewidth=3.0, linestyle="dashed"
)


plt.xticks(np.linspace(daten_cut["index"].min(), daten_cut["index"].max(), 10))
plt.yticks(np.linspace(daten_cut["daten"].min(), daten_cut["daten"].max(), 10))

plt.ylim(daten_cut["daten"].min() - 30)

plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.legend()
plt.tight_layout()
plt.savefig("./build/Caesium-FWHM.pdf")
plt.clf()

# Compton Kontinuum ausschneiden
maske = (caesium_short["index"] >= 1300) & (caesium_short["index"] < 2300)
caesium_compton = caesium_short[maske]
caesium_compton.loc[:, "daten"] = pd.to_numeric(
    caesium_compton["daten"], errors="coerce"
)

# Kanten der Bins für den abgeschnittenen Datensatz
cut_bin_edges = np.arange(
    caesium_compton["index"].min(), caesium_compton["index"].max() + 2
)
bin_centers = (cut_bin_edges[:-1] + cut_bin_edges[1:]) / 2

cost = LeastSquares(
    caesium_compton["index"],
    caesium_compton["daten"],
    np.sqrt(caesium_compton["daten"]),
    linear,
)

# Variante mit LeastSquares
m = Minuit(cost, alpha=10, beta=15)
# Limits?
m.migrad()
m.hesse()

matplotlib.rcParams.update({"font.size": 8})
# Plot der Daten + Fit + Pulls
fig, axs = plt.subplots(
    2,
    sharex=True,
    gridspec_kw={"hspace": 0.05, "height_ratios": [3, 1]},
    layout="constrained",
)
axs[0].errorbar(
    caesium_compton["index"],
    caesium_compton["daten"],
    yerr=np.sqrt(caesium_compton["daten"]),
    fmt="o",
    color="royalblue",
    label=r"$^{137}\mathrm{Cs}$",
    zorder=1,
)
axs[0].plot(
    caesium_compton["index"],
    linear(caesium_compton["index"], *m.values),
    label="fit",
    color="orange",
    linewidth=2.2,
    zorder=2,
)
axs[0].legend()
axs[0].set_ylabel("Counts")

# Chi^2 Test des Fits auf Abbildung schreiben
fit_info = [
    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
]

# Fitparameter auf Abbildung schreiben
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

n_model = linear(caesium_compton["index"], *m.values)
n_error = np.sqrt(n_model)

# Pull berechnen
pull = (caesium_compton["daten"] - n_model) / n_error

axs[1].stairs(pull, cut_bin_edges, fill=True, color="royalblue")
axs[1].axhline(0, color="orange", linewidth=0.8)
axs[1].set_xlabel(r"$\mathrm{Channels}$")
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
plt.savefig("./build/Caesium-Compton-Fit.pdf")
plt.clf()


# Cleanup
plt.close()
peaks.to_csv("./build/peaks_Cs.csv")
