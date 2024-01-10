#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import numpy as np
from scipy.signal import find_peaks
import iminuit
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import norm
from typing import Tuple

matplotlib.rcParams.update({"font.size": 18})

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

plt.figure(figsize=(21, 9), dpi=500)  # TODO andere dpi ausprobieren
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

matplotlib.rcParams.update({"font.size": 8})


def scaled_gauss_cdf(x, s, mu, sigma, plus):
    """
    Eine skalierte kumulative Normalverteilungsfunktion,
    die an die Signale angepasst werden soll.

    Parameter:

    x: array-like, bin edges; muss len(n) + 1 sein (n: Anzahl bins)

    s: Skalierungsparameter der Funktion; ist auch der Integralwert

    mu: Erwartungswert der Normalverteilung

    sigma: Standardabweichung der Normalverteilung
    """
    return s * norm(mu, sigma).cdf(x) + plus * x


def fitmaker_2000(
    abstand_links: int,
    abstand_rechts: int,
    peak_idx: int,
    startwert_schätzen: bool = True,
    peak_width: float = 0,
    peak_position: float = 0,
    peak_height: float = 0,
    limit_s: Tuple = (0, None),
    limit_mu: Tuple = (0, None),
    limit_sigma: Tuple = (0, None),
    zorder1: int = 1,
    zorder2: int = 2,
) -> None:
    """
    Funktion die einen Fit einer skalierbaren Gauß-Funktion vornimmt. Es kann
    eine Schätzung der Startparameter angegeben werden; alternativ wird eine rudimentäre Schätzung durchgeführt.

    Parameter:

    abstand_links: int, gibt an, wo die linke Grenze des zu fittenden Bereichs sein soll

    abstand_rechts: int, gibt an, wo die rechte Grenze des zu fittenden Bereichs sein soll

    peak_idx: int, gibt an, welcher der 13 Peaks gefittet werden soll

    startwert_schätzen: bool, wechselt zwischen automatisch oder manuell geschätzten Startwerten

    peak_width: float, optionaler Schätzwert für sigma, wird nur verwendet wenn startwert_schätzen auf False gesetzt wird

    peak_position: float, optional, wird nur verwendet wenn startwert_schätzen auf False gesetzt wird

    peak_height: float, optionaler Schätzwert für s, wird nur verwendet wenn startwert_schätzen auf False gesetzt wird

    limit_s: Tuple, optionaler (oberer und unterer) Grenzwert s

    limit_mu: Tuple, optionaler (oberer und unterer) Grenzwert mu

    limit_sigma: Tuple, optionaler (oberer und unterer) Grenzwert sigma

    zorder1: int, gibt an auf welcher Ebene die Daten gemalt werden sollen

    zorder2: int, gibt an auf welcher Ebene der Fit gemalt werden soll
    """

    # Bereich um den Peak herum ausschneiden
    maske = (caesium["index"] >= peaks["peaks"][peak_idx] - abstand_links) & (
        caesium["index"] <= peaks["peaks"][peak_idx] + abstand_rechts
    )
    caesium_cut = caesium[maske]

    # Kanten der Bins für den abgeschnittenen Datensatz
    cut_bin_edges = np.arange(
        caesium_cut["index"].min(), caesium_cut["index"].max() + 2
    )

    # Kanten der Bins für den vollständigen Datensatz
    bin_edges = np.arange(caesium["index"].min(), caesium["index"].max() + 2)

    cost = ExtendedBinnedNLL(caesium_cut["daten"], cut_bin_edges, scaled_gauss_cdf)

    if startwert_schätzen == True:
        # Schätze s über die maximale Höhe des Peaks
        peak_height = caesium_cut["daten"].max()

        # Als Startwert für mu: Position des Peaks
        peak_position = peaks["peaks"][peak_idx]

        # Schätze die Breite des Peaks (sigma)
        peak_width = (caesium_cut["index"].max() - caesium_cut["index"].min()) * 0.1

    m = Minuit(cost, s=peak_height, mu=peak_position, sigma=peak_width, plus=20)
    m.limits["s"] = limit_s
    m.limits["mu"] = limit_mu
    m.limits["sigma"] = limit_sigma
    m.limits["plus"] = (15, 20)
    m.migrad()
    m.hesse()

    # So kann man auch irgendwie die Pulls berechnen, weiß nur nicht ganz welche Params da rein
    # müssen
    # print(cost.pulls(*m.values))

    # Plot der Daten + Fit + Pulls
    fig, axs = plt.subplots(
        2,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "height_ratios": [3, 1]},
        layout="constrained",
    )
    axs[0].errorbar(
        caesium_cut["index"],
        caesium_cut["daten"],
        yerr=np.sqrt(caesium_cut["daten"]),
        fmt="o",
        color="royalblue",
        label=r"$^{137}\mathrm{Cs}$" + f"-Peak {peak_idx+1}",
        zorder=zorder1,
    )
    axs[0].stairs(
        np.diff(scaled_gauss_cdf(cut_bin_edges, *m.values)),
        cut_bin_edges,
        label="fit",
        color="orange",
        linewidth=2.2,
        zorder=zorder2,
    )
    axs[0].legend()
    axs[0].set_ylabel("Counts")
    # axs[0].grid(True)

    # Chi^2 Test des Fits auf Abbildung schreiben
    fit_info = [
        f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
    ]

    # Fitparameter auf Abbildung schreiben
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

    n_model = np.diff(scaled_gauss_cdf(cut_bin_edges, *m.values))
    n_error = np.sqrt(n_model)

    # Pull berechnen
    pull = (caesium_cut["daten"] - n_model) / n_error

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
    # axs[1].grid(True)

    axs[0].legend(title="\n".join(fit_info), frameon=False)
    plt.savefig(f"./build/Caesium-Fit-Peak{peak_idx+1}.pdf")
    plt.clf()

    # Linienbreite muss für jeden Peak gespeichert werden
    peaks.loc[peak_idx, "N"] = m.values["s"]
    peaks.loc[peak_idx, "N_err"] = m.errors["s"]


grenzen = pd.DataFrame(data={"L": [150, 15], "R": [150, 15]})
zorder1 = [1, 2]
zorder2 = [2, 1]

for i in range(len(peaks)):
    fitmaker_2000(
        grenzen["L"][i], grenzen["R"][i], i, zorder1=zorder1[i], zorder2=zorder2[i]
    )

peaks.to_csv("./build/peaks_Cs.csv")
