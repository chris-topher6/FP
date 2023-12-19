import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

lamda = 632.99e-9 # wavelength of the laser (meter)

# Determination of maximum contrast

theta, I_max1, I_min1, I_max2, I_min2  = np.genfromtxt("data/contrast.dat", unpack = True)

theta = theta*np.pi/180

def contrast(I_min, I_max):
    return (I_max - I_min)/(I_max + I_min)

K1 = contrast(I_min1, I_max1)
K2 = contrast(I_min2, I_max2)

K_mean = np.mean([K1, K2], axis = 0)
K_std = np.std([K1, K2], axis = 0)

def theo_curve(phi, I_0, delta):
    return I_0 *2*np.abs(np.cos(phi - delta)*np.sin(phi - delta))

params, pcov = op.curve_fit(theo_curve, theta, K_mean, p0 = [1, 0], sigma = K_std)
err = np.sqrt(np.diag(pcov))

print("--------------------------------------------------")
print("Contrast-fit:")
print(f"I_O = {params[0]:.4f} +- {err[0]:.4f}")
print(f"delta / ° = {180*params[1]/np.pi:.4f} +- {180*err[1]/np.pi:.4f}")
print("--------------------------------------------------")


x = np.linspace(-0.1, np.pi + 0.1, 1000)

# Erstellen eines Plots mit zwei Subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Daten und Fit im Hauptplot darstellen
ax1.errorbar(theta, K_mean, yerr=K_std, fmt='.', markersize=5, color='black', ecolor='dimgray', elinewidth=2, capsize=2, label="Data")
ax1.plot(x, theo_curve(x, *params), color='dodgerblue', linestyle='-', linewidth=2, label="Fit")
#ax1.plot(x, theo_curve(x, 0.9,0.05), color='red', linestyle='-', linewidth=2, label="Fit (theory curve)")

# Beschriftungen und Grenzen für den Hauptplot
ax1.set_ylabel(r"$K$", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlim(-0.1, np.pi+0.1)
ax1.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax1.set_xticklabels([0, 45, 90, 135, 180])

# Berechnung der Residuen für den Pull-Plot
pulls = (K_mean - theo_curve(theta, *params)) / K_std
ax2.plot(theta, pulls, 'o', markersize=5, color='red')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)  # Referenzlinie bei 0
ax2.set_ylim(-55, 55)

# Beschriftungen und Grenzen für den Pull-Plot
ax2.set_ylabel("Pulls/$\sigma$")
ax2.set_xlabel(r"$\phi / °$", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Layout anpassen
plt.tight_layout()

# Hintergrundfarbe setzen
fig.patch.set_facecolor('whitesmoke')

# Speichern des Plots in einer PDF-Datei
plt.savefig("build/contrast.pdf", bbox_inches='tight')
plt.close()