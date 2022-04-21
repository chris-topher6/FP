import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp

def gerade(x, m, b):
    return m*x+b

t, K = np.genfromtxt('data/kalibration.dat', unpack=True)

params, covariance_matrix = np.polyfit(K, t, deg=1, cov=True)
uncertainties = np.sqrt(np.diag(covariance_matrix))

#Ausgeben der Parameter
print("\nRegressionsparameter für Kalibration")
errors = np.sqrt(np.diag(covariance_matrix))
for name, value, error in zip('ab', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

x = np.linspace(np.min(K), np.max(K))
plt.plot(x, gerade(x, *params), "k", linewidth=1, label="Regression")
plt.plot(K, t, 'r+', markersize=10, label="Data")
plt.xlabel(r"$K[us]$")
plt.ylabel(r"$t[us]$")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/kalibration.pdf')