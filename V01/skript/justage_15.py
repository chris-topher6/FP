import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp

def gerade(x, m, b):
    return m*x+b

dt, N = np.genfromtxt('data/justage_15.dat', unpack=True)

#fit 
N_cut1  = N[0:8]
dt_cut1 =dt[0:8]
N_cut2  = N[7:23]
dt_cut2 =dt[7:23]
N_cut3  = N[22:27]
dt_cut3 =dt[22:27]

print(dt_cut2[0])
print(dt_cut2[-1])

params1, covariance_matrix1 = np.polyfit(dt_cut1, N_cut1, deg=1, cov=True)
uncertainties1 = np.sqrt(np.diag(covariance_matrix1))
params2, covariance_matrix2 = np.polyfit(dt_cut2, N_cut2, deg=0, cov=True)
uncertainties2 = np.sqrt(np.diag(covariance_matrix2))
params3, covariance_matrix3 = np.polyfit(dt_cut3, N_cut3, deg=1, cov=True)
uncertainties3 = np.sqrt(np.diag(covariance_matrix3))

#Ausgeben der Parameter
print("\nRegressionsparameter für linke Gerade für justage 15")
errors1 = np.sqrt(np.diag(covariance_matrix1))
for name, value, error in zip('ab', params1, errors1):
    print(f'{name} = {value:.8f} ± {error:.8f}')
    
print("\nRegressionsparameter für rechte Gerade für justage 15")
errors3 = np.sqrt(np.diag(covariance_matrix3))
for name, value, error in zip('ab', params3, errors3):
    print(f'{name} = {value:.8f} ± {error:.8f}')

print("\nRegressionsparameter für Platau für justage 15")
errors2 = np.sqrt(np.diag(covariance_matrix2))
for name, value, error in zip('ab', params2, errors2):
    print(f'{name} = {value:.8f} ± {error:.8f}')

halfmax = params2[0]/2
h1 = (halfmax-params1[1])/params1[0]
h3 = (halfmax-params3[1])/params3[0]

x1 = np.linspace(np.min(dt_cut1), np.max(dt_cut1))
x2 = np.linspace(np.min(dt_cut2), np.max(dt_cut2))
y2 = np.ones(len(x2))
x3 = np.linspace(np.min(dt_cut3), np.max(dt_cut3))
xhalf = np.linspace(h1, h3)
yhalf = np.ones(len(xhalf))
print(f"h1={h1}")
print(f"h3={h3}")
plt.plot(x1, params1[0]*x1+params1[1], "orange", linewidth=1, label="Regression")
plt.plot(x2, params2[0]*y2, "darkgreen", linewidth=1, label="Plateau")
plt.plot(xhalf, halfmax*yhalf, "r--", linewidth=1, label="halber Plateauwert")
plt.plot(x3, params3[0]*x3+params3[1], "orange", linewidth=1)
plt.errorbar(dt, N, xerr=0, yerr=np.sqrt(N), color="darkblue", ecolor="royalblue", fmt='.', label="Daten")
#plt.hlines(np.mean(N_cut2)/2, np.min(dt_cut2)-5, np.max(dt_cut2)+6, color='green', linestyles='dashed', label='Plataumittelwert/2')
plt.xlabel(r"$\Delta t_{Delay}[ns]$")
plt.ylabel(r"$N [1/s]$")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/justage_15.pdf')

print(f"Mittelwert=({np.mean(N_cut2):.4}+/-{np.std(N_cut2):.3})")
#print(f"Halbwertsbreite=({np.mean(N_cut2/2):.3}+/-{np.std(N_cut2/2):.3})")