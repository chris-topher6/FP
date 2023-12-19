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

theta, I_min1, I_max1, I_min2, I_max2, I_min3, I_max3 = np.genfromtxt("content/data/contrast.txt", unpack = True)

theta = theta*np.pi/180

def contrast(I_min, I_max):
    return (I_max - I_min)/(I_max + I_min)

K1 = contrast(I_min1, I_max1)
K2 = contrast(I_min2, I_max2)
K3 = contrast(I_min3, I_max3)

K_mean = np.mean([K1, K2, K3], axis = 0)
K_std = np.std([K1, K2, K3], axis = 0)

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

fig, ax = plt.subplots()

ax.errorbar(theta, K_mean, yerr = K_std, lw = 0, marker = ".", ms = 8, color = "black", label = "Data", elinewidth=1, capsize=4)
ax.plot(x, theo_curve(x, *params), c = "firebrick", label = "Fit (theory curve)")
plt.xlim(-0.1, np.pi+0.1)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [0, 45, 90, 135, 180])
plt.ylabel(r"Contrast $K$")
plt.xlabel(r"$\phi \mathbin{/} °$")
plt.ylim(0, 0.7)
plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/contrast.pdf")
plt.close()