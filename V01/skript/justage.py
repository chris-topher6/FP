import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy

#Gerade
def gerade(x, m, b):
    return m*x+b

#string für for loop
data = ['data/justage_10.dat', 'data/justage_15.dat']
pdf  = ['build/justage_10.pdf', 'build/justage_15.pdf']

#Messwerte
dt, N = np.genfromtxt(data[0],   unpack=True)

#in sinnvolle Einheiten
Nu = unumpy.uarray(N, np.sqrt(N))

params,covariance_matrix=np.polyfit(dt, N, deg=1, cov=True)

plt.figure()
plt.errorbar(dt, N, xerr=0, yerr=np.sqrt(N), fmt = "x", ecolor='red', label="data")
x=np.linspace(np.min(dt), np.max(dt))
plt.plot(x, gerade(x, *params), "k", label="Regression")
plt.xlabel(r"$\Delta t [\si{\nano\seconds}]$")
plt.ylabel(r"$N [\si{\per\seconds}]$")
#plt.xlabel("dt")
#plt.ylabel("N")
plt.legend(loc='best')
#plt.tight_layout()
plt.savefig('build/justage_10.pdf')
print("ende")