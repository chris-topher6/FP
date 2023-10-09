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
p_0 = ufloat(1011,1011*0.003) #0.3% Fehler
p_E = ufloat(7.1*10**(-3),7.1*0.1*10**(-3)) #10% Fehler
err = 0.1 #10% Fehler
p_m = (p1+p2+p3)/3
p_sys = np.multiply(p_m,err)
p_stat = np.sqrt(((p1-p_m)**2+(p2-p_m)**2+(p3-p_m)**2)/6)
#np.savetxt('build/Dreh_Evak_Daten.txt', np.column_stack([t, p_m, p_stat, p_sys]), fmt='%6.5f', delimiter=' & ', header='p, p_stat, p_sys', newline='\\\\\n' )
p = unp.uarray(p_m, p_sys)
ln_p=unp.log((p-p_E)/(p_0-p_E))
np.savetxt('build/Dreh_Evak_Daten.txt', np.column_stack([t, p1,p2,p3, p_m, p_stat, p_sys, noms(ln_p), stds(ln_p)]), fmt='%6.5f', delimiter=' & ', header='t, p1,p2,p3, p, p_stat, p_sys, ln, dln', newline='\\\\\n' )
print("t, p1, p2, p3, p, p_stat, p_sys, p_stat, lnp")
p_stat_ufloat=unp.uarray(p_m,p_stat)
p_sys_ufloat=unp.uarray(p_sys,p_stat)
for i in range(len(p_stat_ufloat)):
    print(f"{t[i]} & {p1[i]} & {p2[i]} & {p3[i]} & {p_stat_ufloat[i]} & {p_sys_ufloat[i]} & {ln_p[i]}")
print("\n")

#Fit
t_cut1=t[0:21]
ln_p_cut1=ln_p[0:21]
params1,covariance_matrix1 = curve_fit(gerade, t_cut1, noms(ln_p_cut1), sigma=stds(ln_p_cut1), absolute_sigma=True)
print(f"Die Parameter der Regression im Druckbereich ({p[0]})mbar bis ({p[20]})mbar sind:")
errors1     = np.sqrt(np.diag(covariance_matrix1))
for name, value, error in zip('ab', params1, errors1):
    params1_u = unp.uarray(params1, errors1)
    print(f'{name} = {value:.8f} ± {error:.8f}')

t_cut2=t[21:34]
ln_p_cut2=ln_p[21:34]
#params2,covariance_matrix2 =np.polyfit(t_cut2, noms(ln_p_cut2), deg=1, cov=True)
params2,covariance_matrix2 = curve_fit(gerade, t_cut2, noms(ln_p_cut2), sigma=stds(ln_p_cut2), absolute_sigma=True)
print(f"Die Parameter der Regression im Druckbereich ({p[21]})mbar bis ({p[33]})mbar sind:")
errors2     = np.sqrt(np.diag(covariance_matrix2))
for name, value, error in zip('ab', params2, errors2):
    params2_u = unp.uarray(params2, errors2)
    print(f'{name} = {value:.8f} ± {error:.8f}')

t_cut3=t[34:61]
ln_p_cut3=ln_p[34:61]
#params2,covariance_matrix3 =np.polyfit(t_cut3, noms(ln_p_cut3), deg=1, cov=True)
params3,covariance_matrix3 = curve_fit(gerade, t_cut3, noms(ln_p_cut3), sigma=stds(ln_p_cut3), absolute_sigma=True)
print(f"Die Parameter der Regression im Druckbereich ({p[34]})mbar bis ({p[60]})mbar sind:")
errors3     = np.sqrt(np.diag(covariance_matrix3))
for name, value, error in zip('ab', params3, errors3):
    params3_u = unp.uarray(params3, errors3)
    print(f'{name} = {value:.8f} ± {error:.8f}')

#Plot
plt.figure()
x1 = np.linspace(np.min(t_cut1), np.max(t_cut1))
x2 = np.linspace(np.min(t_cut2), np.max(t_cut2))
x3 = np.linspace(np.min(t_cut3), np.max(t_cut3))
plt.plot(x1, gerade(x1, *params1), label="Regression 1")
plt.plot(x2, gerade(x2, *params2), label="Regression 2")
plt.plot(x3, gerade(x3, *params3), label="Regression 3")
plt.errorbar(t, noms(ln_p), xerr=0.2,     yerr=stds(ln_p),     color='blue', ecolor='red',  markersize=3.5, elinewidth=1, fmt='x', label="Daten")
plt.xlabel(r"$t\,\,[s]$")
plt.ylabel(r"ln$\left[\frac{p-p_E}{p_0-p_E}\right]$")
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
plt.savefig('build/Dreh_Evak.pdf')

print("\nBerechnung des Saugvermögens")
V0 = ufloat(34*10**(-3), 3.4*10**(-3))
S1 = -params1_u[0]*V0
S2 = -params2_u[0]*V0
S3 = -params3_u[0]*V0
print(f"p1={noms((p[0]+p[21])/2):.3}+-{noms((p[0]-p[21])/2):.3}hPa")
print(f"S1={S1*3600}m³/h")
print(f"p2={noms((p[21]+p[34])/2):.3}+-{noms((p[21]-p[34])/2):.3}hPa")
print(f"S2={S2*3600}m³/h")
print(f"p3={noms((p[34]+p[60])/2):.3}+-{noms((p[34]-p[60])/2):.3}hPa")
print(f"S3={S3*3600}m³/h")