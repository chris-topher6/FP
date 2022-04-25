import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

def efunk(x, N0, l):
    return N0*np.exp(-l*x)+0.238

#N = np.genfromtxt('data/ronja.dat', unpack=True)
N = np.genfromtxt('data/myonen.dat', unpack=True)
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
#params, cov = curve_fit(efunk,  noms(t_cut),  N_cut)
params, cov = curve_fit(efunk,  noms(t[4:-57]),  N[4:-57])

print("\nRegressionsparameter für die Lebensdauer sind")
errors = np.sqrt(np.diag(cov))
for name, value, error in zip('Nl', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

#Plot
x=noms(np.linspace(np.min(t[4:-57]), np.max(t[4:-57])))
plt.errorbar(noms(t),     N,     xerr=stds(t),     yerr=np.sqrt(N),     color='red', ecolor='grey',  markersize=3.5, elinewidth=0.5, fmt='.', label="entfernte Daten")
plt.errorbar(noms(t[4:-57]), N[4:-57], xerr=stds(t[4:-57]), yerr=np.sqrt(N[4:-57]), color='navy', ecolor='grey', markersize=3.5, elinewidth=0.5, fmt='.', label="Daten")
plt.plot(x, efunk(x, params[0], params[1]), color='orangered', label="Fit")
plt.xlabel(r"$t [\mu s]$")
plt.ylabel(r"$N [1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('build/lebensdauer.pdf')

#Berechnung der mittleren Lebensdauer
lam=ufloat(params[1], errors[1])
tau=1/lam
print(f"\ndie mittlere Lebensdauer beträgt tau=({tau:.4})us")

tau_pdg=ufloat(2.1969811,0.0000022)
p=(tau-tau_pdg)/tau_pdg *100
print(p)