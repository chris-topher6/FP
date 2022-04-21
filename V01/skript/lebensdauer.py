import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit



def efunk(N0, l, U, x):
    return N0*np.exp(-l*x)+U 

N = np.genfromtxt('data/ronja.dat', unpack=True)
K = np.linspace(1,len(N),len(N))

a = ufloat(0.02166913, 0.00001165)
b = ufloat(0.15088736, 0.00293726)
t = a * K + b

params, cov = curve_fit(efunk,  noms(t),  N, bounds=([75,0.3,0.4],[85,0.46,0.9]))

print(params)

K_null=np.zeros(69,dtype=int)
j=0
for i in range(len(N)):
    if N[i]<1:
        K_null[j]=i+1
        j=j+1
t_null = a * K_null + b

N0 = 82
l  = 0.46
U  = 0.74

x=noms(np.linspace(np.min(t), np.max(t)))
plt.errorbar(noms(t),       N,                     xerr=stds(t),      yerr=np.sqrt(N), color='navy', ecolor='grey', markersize=3.5, elinewidth=0.5, fmt='.', label="Daten")
plt.errorbar(noms(t_null),  np.zeros(len(t_null)), xerr=stds(t_null), yerr=0,          color='red',                 markersize=3.5, elinewidth=0.5, fmt='.', label="entfernte Daten")
plt.plot(x, efunk(N0, l, U, x),                       color='black', label="Ronja")
plt.plot(x, efunk(params[0], params[1], params[2],x), color='orangered', label="fit")
plt.xlabel(r"t[us]")
plt.ylabel(r"$N[1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('build/lebensdauer.pdf')