import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

def efunk(x, N0, l, U):
    return N0*np.exp(-l*x)+U 

N = np.genfromtxt('data/ronja.dat', unpack=True)
K = np.linspace(1,len(N),len(N))

#Kanäle in Zeit umrechnen
a = ufloat(0.02166913, 0.00001165)
b = ufloat(0.15088736, 0.00293726)
t = a * K + b

#cutte unphysikalische Werte (N=0)
indexes=np.zeros(69)
n=0
for i in range(len(N)):
    if N[i]<1:
        indexes[n]=i
        n=n+1
indexes = indexes.astype(int)

N_cut = np.delete(N, indexes)
t_cut = np.delete(t, indexes)

#Regression
params, cov = curve_fit(efunk,  noms(t_cut),  N_cut)

print("\nRegressionsparameter für die Lebensdauer sind")
errors = np.sqrt(np.diag(cov))
for name, value, error in zip('NlU', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

#Plot
x=noms(np.linspace(np.min(t), np.max(t)))
plt.errorbar(noms(t),     N,     xerr=stds(t),     yerr=np.sqrt(N),     color='red', ecolor='grey',  markersize=3.5, elinewidth=0.5, fmt='.', label="entfernte Daten")
plt.errorbar(noms(t_cut), N_cut, xerr=stds(t_cut), yerr=np.sqrt(N_cut), color='navy', ecolor='grey', markersize=3.5, elinewidth=0.5, fmt='.', label="Daten")
plt.plot(x, efunk(x, params[0], params[1], params[2]), color='orangered', label="Fit")
plt.xlabel(r"$t[\mu s]$")
plt.ylabel(r"$N[1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('build/lebensdauer.pdf')

#Berechnung der mittleren Lebensdauer
lam=ufloat(params[1], errors[1])
tau=1/lam
print(f"\ndie mittlere Lebensdauer beträgt tau=({tau})us")