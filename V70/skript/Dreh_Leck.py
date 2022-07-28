import numpy as np 
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

#gerade für die Plots
def gerade(x, m, b):
    return m*x+b

data=[
    "1",
    "2",
    "3",
    "4"
]
cut=[
    2,
    1,
    1,
    1
]
err=np.array([
    0.1,
    0.03,
    0.03,
    0.03
])
p_g_m=np.array([
    0.5,
    10,
    50,
    100
])
p_g_s=np.multiply(p_g_m,err)
p_g = unp.uarray(p_g_m, p_g_s)

for i in range(len(data)):
    print("\nMessung "+data[i])
    t, p1, p2, p3 = np.genfromtxt('data/Dreh_Leck_'+data[i]+'.dat', unpack=True)
    p_m = (p1+p2+p3)/3
    p_sys = p_m*err[i]
    p_stat = np.sqrt(((p1-p_m)**2+(p2-p_m)**2+(p3-p_m)**2)/6)
    np.savetxt('build/Dreh_Leck_'+data[i]+'_Daten.txt', np.column_stack([t, p1,p2,p3, p_m, p_stat, p_sys]), fmt='%6.5f', delimiter=' & ', header='t, p1,p2,p3, p, p_stat, p_sys', newline='\\\\\n' )
    p = unp.uarray(p_m, p_sys)
    print("t, p1, p2, p3, p, p_stat, p_sys, p_stat")
    p_stat_ufloat=unp.uarray(p_m,p_stat)
    p_sys_ufloat=unp.uarray(p_sys,p_stat)
    for j in range(len(p_stat_ufloat)):
        print(f"{t[j]} & {p1[j]} & {p2[j]} & {p3[j]} & {p_stat_ufloat[j]} & {p_sys_ufloat[j]} \\\\")
    print("test")
    #Fit
    t_cut1=t[cut[i]:]
    p_cut1=p[cut[i]:]
#    params1,covariance_matrix1 =np.polyfit(t_cut1, noms(p_cut1), deg=1, cov=True)
    params1,covariance_matrix1 = curve_fit(gerade, t_cut1, noms(p_cut1), sigma=stds(p_cut1), absolute_sigma=True)
    print("Die Parameter der ersten Regression sind:")
    errors1     = np.sqrt(np.diag(covariance_matrix1))
    for name, value, error in zip('ab', params1, errors1):
        params1_u = unp.uarray(params1, errors1)
        print(f'{name} = {value:.8f} ± {error:.8f}')

    #Plot
    plt.figure()
    x1 = np.linspace(np.min(t_cut1), np.max(t_cut1))
    plt.plot(x1, gerade(x1, *params1), "k", label="Regression")
    plt.errorbar(t, noms(p), xerr=0.2,     yerr=stds(p),     color='blue', ecolor='red',  markersize=3.5, elinewidth=1, fmt='x', label="Daten")
    plt.xlabel(r"$t\,\,[s]$")
    plt.ylabel(r"$p\,\,[hPa]$")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.savefig('build/Dreh_Leck_'+data[i]+'.pdf')

    #print("\nBerechnung des Saugvermögens")
    V0  = ufloat(34*10**(-3), 3.4*10**(-3))
    S1 = params1_u[0]*V0/p_g[i]
    print(f"p1={noms((p_cut1[0]+p_cut1[-1])/2):.4}+-{noms((p_cut1[0]-p_cut1[-1])/2):.4}hPa")
    print(f"S1={S1*3600}m³/h")