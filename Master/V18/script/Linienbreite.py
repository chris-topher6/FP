#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import iminuit
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import sqrt
from scipy.stats import norm
from typing import Tuple, Callable, Any


matplotlib.rcParams.update({"font.size": 18})
# plt.rcParams["figure.constrained_layout.use"] = True

SKIP_ANFANG = 12
SKIP_ENDE = 14

# Einlesen der Kalibrations- und Untergrundmessung
europium = pd.read_csv("./data/Eu1.Spe", skiprows=SKIP_ANFANG, header=None)
europium.columns = ["daten"]
untergrund = pd.read_csv("./data/Untergrund.Spe", skiprows=SKIP_ANFANG, header=None)
untergrund.columns = ["daten"]

# Rotz am Ende entfernen
europium = europium.iloc[:-SKIP_ENDE]
untergrund = untergrund.iloc[:-SKIP_ENDE]

# Sicherstellen, dass die Spalte "data" numerische Werte enthält
europium["daten"] = pd.to_numeric(europium["daten"], errors="coerce")
untergrund["daten"] = pd.to_numeric(untergrund["daten"], errors="coerce")

# Index hinzufügen
europium["index"] = europium.index
untergrund["index"] = untergrund.index

# Plot der Untergrundmessung
plt.figure(figsize=(21, 9))

plt.bar(
    untergrund["index"],
    untergrund["daten"],
    linewidth=2,
    width=1.1,
    color="orange",
    label="Background",
)

plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(europium["daten"].min(), europium["daten"].max(), 10))

plt.ylim(europium["daten"].min() - 30)

plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Untergrund.pdf")
plt.clf()

# Untergrundmessung dauerte 76353s; Eu Messung dauerte 2804s; normiere auf
# letzteres für Plot beider Messungen
untergrund["daten"] = untergrund["daten"] * (2804 / 76353)

# Plot der beiden Messungen in einem
plt.figure(figsize=(21, 9))

plt.bar(
    europium["index"],
    europium["daten"],
    linewidth=2,
    width=1.1,
    label=r"$^{152}\mathrm{Eu}$",
    color="royalblue",
)
plt.bar(
    untergrund["index"],
    untergrund["daten"],
    linewidth=2,
    width=1.1,
    color="orange",
    label="Background",
)

plt.xticks(np.linspace(0, 8191, 10))
plt.yticks(np.linspace(europium["daten"].min(), europium["daten"].max(), 10))

plt.ylim(europium["daten"].min() - 30)

plt.xlabel(r"Channels")
plt.ylabel(r"Signals")

plt.grid(True, linewidth=0.1)
plt.legend()
plt.tight_layout()
plt.savefig("./build/Europium-Untergrund.pdf")
plt.clf()

matplotlib.rcParams.update({"font.size": 8})

# Untergrund entfernen
europium["daten"] = europium["daten"] - untergrund["daten"]

# Negative Werte in einem Histogramm sind unphysikalisch
europium["daten"] = europium["daten"].clip(lower=0)

# Vorher bestimmte Peaks einlesen
peaks = pd.read_csv("./build/peaks.csv")

# Schonmal neue Spalten anlegen
peaks["N"] = float(0)
peaks["N_err"] = float(0)


def scaled_gauss_cdf(x, s, mu, sigma):
    """
    Eine skalierte kumulative Normalverteilungsfunktion,
    die an die Signale angepasst werden soll.

    Parameter:

    x: array-like, bin edges; muss len(n) + 1 sein (n: Anzahl bins)

    s: Skalierungsparameter der Funktion; ist auch der Integralwert

    mu: Erwartungswert der Normalverteilung

    sigma: Standardabweichung der Normalverteilung
    """
    return s * norm(mu, sigma).cdf(x)


def fitmaker_2000(
    daten_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    abstand_links: int,
    abstand_rechts: int,
    peak_idx: int,
    fit_function: Callable[..., Any],
    startwert_schätzen: bool = True,
    peak_width: float = 0,
    peak_position: float = 0,
    peak_height: float = 0,
    limit_s: Tuple = (0, None),
    limit_mu: Tuple = (0, None),
    limit_sigma: Tuple = (0, None),
    zorder1: int = None,
    zorder2: int = None,
) -> None:
    """
    Funktion die einen Fit einer skalierbaren Gauß-Funktion vornimmt. Es kann
    eine Schätzung der Startparameter angegeben werden; alternativ wird eine rudimentäre Schätzung durchgeführt.

    Parameter:
    daten_df: Pandas Dataframe, enthält die Daten

    peaks_df: Pandas Dataframe, enthält die Peaks

    abstand_links: int, gibt an, wo die linke Grenze des zu fittenden Bereichs sein soll

    abstand_rechts: int, gibt an, wo die rechte Grenze des zu fittenden Bereichs sein soll

    peak_idx: int, gibt an, welcher der 13 Peaks gefittet werden soll

    fit_function: Callable, Verteilung die gefittet werden soll

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
    maske = (daten_df["index"] >= peaks_df["peaks"][peak_idx] - abstand_links) & (
        daten_df["index"] <= peaks_df["peaks"][peak_idx] + abstand_rechts
    )
    daten_cut = daten_df[maske]

    # Kanten der Bins für den abgeschnittenen Datensatz
    cut_bin_edges = np.arange(daten_cut["index"].min(), daten_cut["index"].max() + 2)
    bin_centers = (cut_bin_edges[:-1] + cut_bin_edges[1:]) / 2

    # Kanten der Bins für den vollständigen Datensatz
    bin_edges = np.arange(daten_df["index"].min(), daten_df["index"].max() + 2)

    cost = ExtendedBinnedNLL(daten_cut["daten"], cut_bin_edges, fit_function)

    if startwert_schätzen == True:
        # Schätze s über die maximale Höhe des Peaks
        peak_height = daten_cut["daten"].max()

        # Als Startwert für mu: Position des Peaks
        peak_position = peaks_df["peaks"][peak_idx]

        # Schätze die Breite des Peaks (sigma)
        peak_width = (daten_cut["index"].max() - daten_cut["index"].min()) * 0.1

    if zorder1 is not None and zorder2 is not None:
        m = Minuit(cost, s=peak_height, mu=peak_position, sigma=peak_width, b=20)
        m.limits["b"] = (15, 20)
    else:
        m = Minuit(cost, s=peak_height, mu=peak_position, sigma=peak_width)

    m.limits["s"] = limit_s
    m.limits["mu"] = limit_mu
    m.limits["sigma"] = limit_sigma
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
    if zorder1 is not None and zorder2 is not None:
        axs[0].errorbar(
            bin_centers,
            daten_cut["daten"],
            yerr=np.sqrt(daten_cut["daten"]),
            fmt="o",
            color="royalblue",
            label=r"$^{137}\mathrm{Cs}$"
            + f"-Peak {peak_idx+1}",  # Hier müsste man nochmal das Label genereller machen
            zorder=zorder1,
            barsabove=True,
        )
        axs[0].stairs(
            np.diff(fit_function(cut_bin_edges, *m.values)),
            cut_bin_edges,
            label="fit",
            color="orange",
            linewidth=2.2,
            zorder=zorder2,
        )
    else:
        axs[0].errorbar(
            bin_centers,
            daten_cut["daten"],
            yerr=np.sqrt(daten_cut["daten"]),
            fmt="o",
            color="royalblue",
            label=r"$^{152}\mathrm{Eu}$"
            + f"-Peak {peak_idx+1}",  # Hier müsste man nochmal das Label genereller machen
            barsabove=True,
        )
        axs[0].stairs(
            np.diff(fit_function(cut_bin_edges, *m.values)),
            cut_bin_edges,
            label="fit",
            color="orange",
            linewidth=2.2,
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

    n_model = np.diff(fit_function(cut_bin_edges, *m.values))
    n_error = np.sqrt(n_model)

    # Pull berechnen
    pull = (daten_cut["daten"] - n_model) / n_error

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
    if zorder1 is not None and zorder2 is not None:
        plt.savefig(f"./build/Caesium-Fit-Peak{peak_idx+1}.pdf")
    else:
        plt.savefig(
            f"./build/Europium-Fit-Peak{peak_idx+1}.pdf"
        )  # Bennenung genereller machen
    plt.clf()

    # Linienbreite muss für jeden Peak gespeichert werden
    peaks.loc[peak_idx, "N"] = m.values["s"]
    peaks.loc[peak_idx, "N_err"] = m.errors["s"]
    return None


grenzen = {
    "L": [5, 8, 8, 10, 12, 8, 11, 12, 12, 15, 12, 15, 18],
    "R": [12, 12, 7, 8, 12, 9, 15, 12, 10, 15, 16, 15, 12],
}
grenzen = pd.DataFrame(data=grenzen)

for i in range(len(peaks)):
    fitmaker_2000(
        europium, peaks, grenzen["L"][i], grenzen["R"][i], i, scaled_gauss_cdf
    )

plt.close()
peaks.to_csv("./build/peaks.csv", index=False)
