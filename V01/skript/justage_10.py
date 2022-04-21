import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp

def gerade(x, m, b):
    return m*x+b

dt, N = np.genfromtxt('data/justage_10.dat', unpack=True)

#fit 
N_cut=N[10:21]
dt_cut = dt[10:21]

params, covariance_matrix = np.polyfit(dt_cut, N_cut, deg=1, cov=True)
uncertainties = np.sqrt(np.diag(covariance_matrix))

#Ausgeben der Parameter
print("\nRegressionsparameter für Platau für justage 15")
errors = np.sqrt(np.diag(covariance_matrix))
for name, value, error in zip('ab', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

x = np.linspace(np.min(dt_cut), np.max(dt_cut))
plt.plot(x, gerade(x, *params), "k", linewidth=1, label="Regression")
plt.errorbar(dt, N, xerr=0, yerr=np.sqrt(N), fmt='.', label="Data")
#plt.hlines(np.mean(N_cut)/2, np.min(dt_cut)-5, np.max(dt_cut)+6, color='green', linestyles='dashed', label='Plataumittelwert/2')
plt.xlabel(r"$dt[ns]$")
plt.ylabel(r"$N[1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/justage_10.pdf')