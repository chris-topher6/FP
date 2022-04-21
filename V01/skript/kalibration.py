import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp

def gerade(x, m, b):
    return m*x+b

t, K = np.genfromtxt('data/kalibration.dat', unpack=True)

params, covariance_matrix = np.polyfit(t, K, deg=1, cov=True)
uncertainties = np.sqrt(np.diag(covariance_matrix))

#Ausgeben der Parameter
print("\nRegressionsparameter für Kalibration")
errors = np.sqrt(np.diag(covariance_matrix))
for name, value, error in zip('ab', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

x = np.linspace(np.min(t), np.max(t))
plt.plot(x, gerade(x, *params), "k", linewidth=1, label="Regression")
plt.plot(t, K, '.', label="Data")
plt.xlabel(r"$t[us]$")
plt.ylabel(r"K")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/kalibration.pdf')