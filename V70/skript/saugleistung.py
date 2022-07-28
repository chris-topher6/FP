
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit
import scipy.constants as const
import sympy
import os
from tabulate import tabulate           # falls nicht installiert "pip install tabulate"
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

#Drehschieberpumpe
t_1 =np.linspace(0,650,1000)
plt.figure()
plt.fill_between(t_1, 4.6, 5.5, label="Herstellerangaben")
plt.errorbar(2.68, 2.5, xerr = np.sqrt(2)*0.983,  yerr=0.4, fmt='.', label = "Leck 0.4 hPa")
plt.errorbar(54.2, 4.6, xerr = np.sqrt(2)*36.5,   yerr=0.5, fmt='.', label = "Leck 10 hPa")
plt.errorbar(237,  4.3, xerr = np.sqrt(2)*171,    yerr=0.5, fmt='.', label = "Leck 50 hPa")
plt.errorbar(290,  4.1, xerr = np.sqrt(2)*160,    yerr=0.4, fmt='.', label = "Leck 100 hPa")
plt.errorbar(325,  3.8, xerr = np.sqrt(2)*322,    yerr=0.4, fmt='.', label = "Evakuierung 1")
plt.errorbar(0.255,0.48,xerr = np.sqrt(2)*0.095,  yerr=0.08,fmt='.', label = "Evakuierung 2")
plt.xscale('log')
plt.xlabel(r"$p$ $ [hPa]$")
plt.ylabel(r"$S$ $ [\frac{m³}{h}]$")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig("build/saug_dreh.pdf")



#Turbomolekularpumpe
t_1 =np.linspace(0,25,1000)
plt.figure()
plt.hlines(277.2, 0, 25, label="Herstellerangaben")
plt.errorbar(0.21, 25,   xerr = np.sqrt(2)*0.05,    yerr=9,    fmt='.', label = "Leck1: 5e-5hPa")
plt.errorbar(1.07, 38,   xerr = np.sqrt(2)*0.25,    yerr=16,   fmt='.', label = "Leck2: 5e-5hPa")
plt.errorbar(0.32, 25,   xerr = np.sqrt(2)*0.09,    yerr=8,    fmt='.', label = "Leck1: 7e-5hPa")
plt.errorbar(1.8,  46,   xerr = np.sqrt(2)*0.4,     yerr=18,   fmt='.', label = "Leck2: 7e-5hPa")
plt.errorbar(0.49, 28,   xerr = np.sqrt(2)*0.13,    yerr=10,   fmt='.', label = "Leck1: 1e-4hPa")
plt.errorbar(3.5 , 67,   xerr = np.sqrt(2)*0.8,     yerr=26,   fmt='.', label = "Leck2: 1e-4hPa")
plt.errorbar(4.05, 97.7, xerr = np.sqrt(2)*2.88,    yerr=35.1, fmt='.', label = "Leck1: 2e-4hPa")
plt.errorbar(14.1, 96.8, xerr = np.sqrt(2)*5.89,    yerr=42.3, fmt='.', label = "Leck2: 2e-4hPa")
plt.errorbar(2.71,   58, xerr = np.sqrt(2)*2.42,    yerr=13,   fmt='.', label = "Evakuierung 1")
plt.errorbar(0.167,  22, xerr = np.sqrt(2)*0.131,   yerr=6,    fmt='.', label = "Evakuierung 2")
plt.errorbar(0.0186, 1.2,xerr = np.sqrt(2)*0.00507, yerr=0.8,  fmt='.', label = "Evakuierung 3")

plt.xscale('log')
plt.xlabel(r"$p$ $ [10^{-3}hPa]$")
plt.ylabel(r"$S$ $ [\frac{m³}{h}]$")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig("build/saug_turbo.pdf")