
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
plt.errorbar(2.68, 2.5, xerr = 0.983,  yerr=0.4, fmt='.', label = "Leck 0.4 hPa")
plt.errorbar(54.2, 4.6, xerr = 36.5,  yerr=0.5, fmt='.', label = "Leck 10 hPa")
plt.errorbar(237,  4.3, xerr = 171,    yerr=0.5, fmt='.', label = "Leck 50 hPa")
plt.errorbar(290,  4.1, xerr = 160,    yerr=0.4, fmt='.', label = "Leck 100 hPa")
plt.errorbar(325,  3.8, xerr = 322,  yerr=0.4, fmt='.', label = "Evakuierung 1")
plt.errorbar(0.255,0.48,xerr = 0.095,yerr=0.08,fmt='.', label = "Evakuierung 2")
plt.xscale('log')
plt.xlabel(r"$p$ $ [hPa]$")
plt.ylabel(r"$S$ $ [\frac{m³}{h}]$")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig("build/saug_dreh.pdf")



#Turbomolekularpumpe
t_1 =np.linspace(0,650,1000)
plt.figure()
plt.hlines(277.2, 0, 650, label="Herstellerangaben")
plt.errorbar(0.21, 25,   xerr = 0.05,    yerr=9,    fmt='.', label = "Leck1: 5e-5hPa")
plt.errorbar(1.07, 38,   xerr = 0.25,    yerr=16,   fmt='.', label = "Leck2: 5e-5hPa")
plt.errorbar(0.32, 25,   xerr = 0.09,    yerr=8,    fmt='.', label = "Leck1: 7e-5hPa")
plt.errorbar(1.8,  46,   xerr = 0.4,     yerr=18,   fmt='.', label = "Leck2: 7e-5hPa")
plt.errorbar(0.49, 28,   xerr = 0.13,    yerr=10,   fmt='.', label = "Leck1: 1e-4hPa")
plt.errorbar(3.5 , 67,   xerr = 0.8,     yerr=26,   fmt='.', label = "Leck2: 1e-4hPa")
plt.errorbar(4.05, 97.7, xerr = 2.88,    yerr=35.1, fmt='.', label = "Leck1: 2e-4hPa")
plt.errorbar(14.1, 96.8, xerr = 5.89,    yerr=42.3, fmt='.', label = "Leck2: 2e-4hPa")
plt.errorbar(2.71,   58, xerr = 2.42,    yerr=13,   fmt='.', label = "Evakuierung 1")
plt.errorbar(0.167,  22, xerr = 0.131,   yerr=6,    fmt='.', label = "Evakuierung 2")
plt.errorbar(0.0186, 1.2,xerr = 0.00507, yerr=0.8,  fmt='.', label = "Evakuierung 3")

plt.xscale('log')
plt.xlabel(r"$p$ $ [mhPa]$")
plt.ylabel(r"$S$ $ [\frac{m³}{h}]$")
plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig("build/saug_turbo.pdf")


#t_2 =np.linspace(10**(-3),0.01,1000)
#
#turbo_theo = np.ones(1000) * 77
#
#plt.figure()
#plt.rc('legend', fontsize= 6)
##plt.plot(t_2, turbo_theo, label = "Theoriewert" )
#plt.errorbar((turbo_leck_1_1[0]+turbo_leck_1_1[-1])/2, noms(Saug_turbo_1), xerr = 0, yerr=stds(Saug_turbo_1), fmt='x', label = "Leck 1e-4 mbar")
#plt.errorbar((turbo_leck_2_1[0]+turbo_leck_2_1[-1])/2, noms(Saug_turbo_2), xerr = 0, yerr=stds(Saug_turbo_2), fmt='x', label = "Leck 2e-4 mbar")
#plt.errorbar((turbo_leck_3_1[0]+turbo_leck_3_1[-1])/2, noms(Saug_turbo_7), xerr = 0, yerr=stds(Saug_turbo_7), fmt='x', label = "Leck 7e-4 mbar")
#plt.errorbar((turbo_leck_4_1[0]+turbo_leck_4_1[-1])/2, noms(Saug_turbo_5), xerr = 0, yerr=stds(Saug_turbo_5), fmt='x', label = "Leck 5e-4 mbar")
#plt.errorbar((turbo_pump_p_1[0 ]+ turbo_pump_p_1[4])/2, noms(Saug_turbo_p_pump[0]),             xerr = (turbo_pump_p_1[0]-turbo_pump_p_1[4]          )/2, yerr=stds(Saug_turbo_p_pump[0]), fmt='x', label = "Evakuierung Pumpe 1")
#plt.errorbar((turbo_pump_p_1[4]+  turbo_pump_p_1[9])/2, noms(Saug_turbo_p_pump[1]),             xerr = (turbo_pump_p_1[4]-turbo_pump_p_1[9]          )/2, yerr=stds(Saug_turbo_p_pump[1]), fmt='x', label = "Evakuierung Pumpe 2")
#plt.errorbar((turbo_pump_p_1[9]+  turbo_pump_p_1[-1])/2, noms(Saug_turbo_p_pump[2]),            xerr = (turbo_pump_p_1[9]-turbo_pump_p_1[-1]         )/2, yerr=stds(Saug_turbo_p_pump[2]), fmt='x', label = "Evakuierung Pumpe 3")
#plt.errorbar((turbo_vent_p_1[0 ]+ turbo_vent_p_1[4])/2, noms(Saug_turbo_p_vent[0]),             xerr = (turbo_vent_p_1[0]-turbo_vent_p_1[4]          )/2, yerr=stds(Saug_turbo_p_vent[0]), fmt='x', label = "Evakuierung Ventil 1")
#plt.errorbar((turbo_vent_p_1[4]+  turbo_vent_p_1[9])/2,  noms(Saug_turbo_p_vent[1]),            xerr = (turbo_vent_p_1[4]-turbo_vent_p_1[9]          )/2, yerr=stds(Saug_turbo_p_vent[1]), fmt='x', label = "Evakuierung Ventil 2")
#plt.errorbar((turbo_vent_p_1[9]+  turbo_vent_p_1[-1])/2, noms(Saug_turbo_p_vent[2]),            xerr = (turbo_vent_p_1[9]-turbo_vent_p_1[-1]         )/2, yerr=stds(Saug_turbo_p_vent[2]), fmt='x', label = "Evakuierung Ventil 3")
#plt.xscale('log')
#plt.rc('axes', labelsize= size_label)
#plt.xlabel(r"$p$ $ [mbar]$")
#plt.ylabel(r"$S$ $ [\frac{l}{s}]$")
###plt.xticks([5*10**3,10**4,2*10**4,4*10**4],[r"$5*10^3$", r"$10^4$", r"$2*10^4$", r"$4*10^4$"])
###plt.yticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],[r"$0$",r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$",r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"])
#plt.tight_layout()
#plt.legend(loc = 'best')
#plt.savefig("build/saug_turbo.pdf")