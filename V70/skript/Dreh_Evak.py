import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

#gerade für die Plots
def gerade(x, m, b):
    return m*x+b

t, p1, p2, p3 = np.genfromtxt('data/Dreh_Evak.dat', unpack=True)
p_0 = ufloat(1011,101.1)
p_E = ufloat(7.1*10**(-3),7.1*0.03*10**(-3))
err = 0.1
p_m = (p1+p2+p3)/3
p_sys = np.multiply(p_m,err)
p_stat = np.sqrt(((p1-p_m)**2+(p2-p_m)**2+(p3-p_m)**2)/2)
print(f"statistischer Fehler\n{p_stat}")
print(f"systematischer Fehler\n{p_sys}")
p = unp.uarray(p_m, p_sys)
ln_p=unp.log((p-p_E)/(p_0-p_E))

#Fit
t_cut1=t[1:20]
ln_p_cut1=ln_p[1:20]
params1,covariance_matrix1 =np.polyfit(t_cut1, noms(ln_p_cut1), deg=1, cov=True)
print("Die Parameter der ersten Regression sind:")
errors1     = np.sqrt(np.diag(covariance_matrix1))
for name, value, error in zip('ab', params1, errors1):
    params1_u = unp.uarray(params1, errors1)
    print(f'{name} = {value:.8f} ± {error:.8f}')

t_cut2=t[40:62]
ln_p_cut2=ln_p[40:62]
params2,covariance_matrix2 =np.polyfit(t_cut2, noms(ln_p_cut2), deg=1, cov=True)
print("Die Parameter der zweiten Regression sind:")
errors2     = np.sqrt(np.diag(covariance_matrix2))
for name, value, error in zip('ab', params2, errors2):
    params2_u = unp.uarray(params2, errors2)
    print(f'{name} = {value:.8f} ± {error:.8f}')

#Plot
plt.figure()
x1 = np.linspace(np.min(t_cut1), np.max(t_cut1))
x2 = np.linspace(np.min(t_cut2), np.max(t_cut2))
plt.plot(x1, gerade(x1, *params1), "k", label="Regression")
plt.plot(x2, gerade(x2, *params2), "k")
plt.errorbar(t, noms(ln_p), xerr=0.2,     yerr=stds(ln_p),     color='red', ecolor='grey',  markersize=3.5, elinewidth=0.5, fmt='.', label="Daten")
plt.xlabel(r"$t [s]$")
plt.ylabel(r"ln$\left[\frac{p-p_E}{p_0-p_E}\right]$")
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('build/Dreh_Evak.pdf')

print("\nBerechnung des Saugvermögens")
V0 = ufloat(34*10**(-3), 3.4*10**(-3))
S1 = -params1_u[0]*V0
S2 = -params2_u[0]*V0
print(f"S1={S1*3600}m³/h")
print(f"S2={S2*3600}m³/h")