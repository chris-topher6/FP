import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

def gerade(x, m, b):
    return m*x+b

dt, N = np.genfromtxt('data/justage_10.dat', unpack=True)

#fit 
N_cut1 = N[0:5]
dt_cut1=dt[0:5]
N_cut2 = N[4:21]
dt_cut2=dt[4:21]
N_cut3 = N[20:26]
dt_cut3=dt[20:26]

print(dt_cut2[0])
print(dt_cut2[-1])

params1, covariance_matrix1 = np.polyfit(dt_cut1, N_cut1, deg=1, cov=True)
uncertainties1 = np.sqrt(np.diag(covariance_matrix1))
params2, covariance_matrix2 = np.polyfit(dt_cut2, N_cut2, deg=0, cov=True)
uncertainties2 = np.sqrt(np.diag(covariance_matrix2))
params3, covariance_matrix3 = np.polyfit(dt_cut3, N_cut3, deg=1, cov=True)
uncertainties3 = np.sqrt(np.diag(covariance_matrix3))

#Ausgeben der Parameter
print("\nRegressionsparameter für linke Gerade für justage 10")
errors1 = np.sqrt(np.diag(covariance_matrix1))
for name, value, error in zip('ab', params1, errors1):
    print(f'{name} = {value:.8f} ± {error:.8f}')

print("\nRegressionsparameter für rechte Gerade für justage 10")
errors3 = np.sqrt(np.diag(covariance_matrix3))
for name, value, error in zip('ab', params3, errors3):
    print(f'{name} = {value:.8f} ± {error:.8f}')

print("\nRegressionsparameter für Platau für justage 10")
errors2 = np.sqrt(np.diag(covariance_matrix2))
for name, value, error in zip('ab', params2, errors2):
    print(f'{name} = {value:.8f} ± {error:.8f}')

halfmax = params2[0]/2
params1_err= unp.uarray(params1,errors1)
params2_err= unp.uarray(params2,errors2)
params3_err= unp.uarray(params3,errors3)
h1 = (halfmax-params1_err[1])/params1_err[0]
h3 = (halfmax-params3_err[1])/params3_err[0]
x1 = np.linspace(np.min(dt_cut1), np.max(dt_cut1))
x2 = np.linspace(np.min(dt_cut2), np.max(dt_cut2))
x3 = np.linspace(np.min(dt_cut3), np.max(dt_cut3))
y2 = np.ones(len(x2))
xhalf = np.linspace(noms(h1), noms(h3))
yhalf = np.ones(len(xhalf))
h_theo=ufloat(20,0)
p=(h_theo-h3+h1)/(h_theo)*100
print(f"h1={h1}")
print(f"h3={h3}")
print(f"Halbwertsbreite={h3-h1}")
print(f"Abweichung p={p}%")
plt.plot(x1, params1[0]*x1+params1[1], "orange", linewidth=1, label="Regression")
plt.plot(x2, params2[0]*y2, "darkgreen", linewidth=1, label="Plateau")
plt.plot(xhalf, halfmax*yhalf, "r--", linewidth=1, label="halber Plateauwert")
plt.plot(x3, params3[0]*x3+params3[1], "orange", linewidth=1)
plt.errorbar(dt, N, xerr=0, yerr=np.sqrt(N), color="darkblue", ecolor="royalblue", fmt='.', label="Daten")
#plt.hlines(np.mean(N_cut2)/2, np.min(dt_cut2)-5, np.max(dt_cut2)+6, color='green', linestyles='dashed', label='Plataumittelwert/2')
plt.xlabel(r"$\Delta t_{Delay} [ns]$")
plt.ylabel(r"$N [1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/justage_10.pdf')

print(f"Mittelwert=({np.mean(N_cut2):.4}+/-{np.std(N_cut2):.3})")
#print(f"Halbwertsbreite=({np.mean(N_cut2/2):.3}+/-{np.std(N_cut2/2):.3})")