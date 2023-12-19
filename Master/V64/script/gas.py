import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

lamda = 632.99e-9 # wavelength of the laser (meter)

# Refractive index of air

p, m1, m2, m3, m4, m5 = np.genfromtxt("data/gas.dat", unpack = True)

T_0 = 273.15 + 21.7 # K Raumtemperatur
L = ufloat(0.1, 0.1e-3)

def n_air_exp(M):
    return M*lamda/(L) + 1

M = unp.uarray(np.mean([m1, m2, m3, m4, m5], axis = 0), np.std([m1, m2, m3, m4, m5], axis = 0))

n = n_air_exp(M)
print("---------------------------------------------------")
print("Brechungsindices Luft:")
for i in range(len(n)):
    print(f"{n[i]:.7f}")
print("---------------------------------------------------")

def n_air_theo(p, A, b):
    return (3/2) * A * p/(const.R *T_0) + b

params2, pcov2 = op.curve_fit(n_air_theo, p, noms(n))   
err2 = np.sqrt(np.diag(pcov2))

n_air_exp = 3/2 * ufloat(params2[0], err2[0])*(1013)/(const.R *(273.15 + 22.5)) + ufloat(params2[1], err2[1])

print("--------------------------------------------------")
print("Fit: Refractive Index of Air:")
print(f"A = {params2[0]:.4e} +- {err2[0]:.4e}")
print(f"b = {params2[1]:.8f} +- {err2[1]:.8f}")
print(f"Experimental Value: n = {n_air_exp}")
print("--------------------------------------------------")

x = np.linspace(0, 1000, 100)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Daten und Fit im Hauptplot darstellen
ax1.errorbar(p, noms(n), yerr=stds(n), fmt='.', markersize=5, color='black', ecolor='dimgray', elinewidth=2, capsize=2, label="Data")
ax1.plot(x, n_air_theo(x, *params2), color='dodgerblue', linestyle='-', linewidth=2, label="Fit")

# Beschriftungen und Grenzen für den Hauptplot
ax1.set_ylabel(r"$n$", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# Berechnung der Residuen für den Pull-Plot
pulls = (noms(n) - n_air_theo(p, *params2)) / stds(n)
ax2.plot(p, pulls, 'o', markersize=5, color='red')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)  # Referenzlinie bei 0
ax2.set_ylim(-1.5, 1.5)

# Beschriftungen und Grenzen für den Pull-Plot
ax2.set_ylabel("Pulls/$\sigma$")
ax2.set_xlabel(r"$p\ $mb", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Layout anpassen
plt.tight_layout()

# Hintergrundfarbe setzen
fig.patch.set_facecolor('whitesmoke')

# Speichern des Plots in einer PDF-Datei
plt.savefig("build/n_air.pdf", bbox_inches='tight')
# Schließen des Plots, um Speicher freizugeben
plt.close()
