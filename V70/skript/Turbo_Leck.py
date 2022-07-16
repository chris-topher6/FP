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
cut1=[
    16,
    17,
    16,
    19
]
cut2=[
    21,
    19,
    19,
    19
]
cut3=[
    34,
    36,
    34,
    36
]
err=0.3
p_g_m=[0.05, 0.07, 0.1, 0.2]
p_g_s=np.multiply(p_g_m,err)
p_g = unp.uarray(p_g_m, p_g_s)

for i in range(len(data)-1):
    print("\nMessung "+data[i])
    t, p1, p2, p3 = np.genfromtxt('data/Turbo_Leck_'+data[i]+'.dat', unpack=True)
    p_m = (p1+p2+p3)/3
    p_sys = np.multiply(p_m,err)
    p_stat = np.sqrt(((p1-p_m)**2+(p2-p_m)**2+(p3-p_m)**2)/2)
    np.savetxt('build/Turbo_Leck_'+data[i]+'_Daten.txt', np.column_stack([t, p_m, p_stat, p_sys]), fmt='%6.5f', delimiter=' & ', header='p, p_stat, p_sys', newline='\\\\\n' )
    p = unp.uarray(p_m, p_sys)

    #Fit
    t_cut1=t[:cut1[i]]
    p_cut1=p[:cut1[i]]
    params1,covariance_matrix1 = curve_fit(gerade, t_cut1, noms(p_cut1), sigma=stds(p_cut1), absolute_sigma=True)
    print("Die Parameter der ersten Regression sind:")
    errors1     = np.sqrt(np.diag(covariance_matrix1))
    for name, value, error in zip('ab', params1, errors1):
        params1_u = unp.uarray(params1, errors1)
        print(f'{name} = {value:.8f} ± {error:.8f}')

    t_cut2=t[cut2[i]:cut3[i]]
    p_cut2=p[cut2[i]:cut3[i]]
    params2,covariance_matrix2 = curve_fit(gerade, t_cut2, noms(p_cut2), sigma=stds(p_cut2), absolute_sigma=True)
    print("Die Parameter der zweiten Regression sind:")
    errors2     = np.sqrt(np.diag(covariance_matrix2))
    for name, value, error in zip('ab', params2, errors2):
        params2_u = unp.uarray(params2, errors2)
        print(f'{name} = {value:.8f} ± {error:.8f}')


    #Plot
    plt.figure()
    x1 = np.linspace(np.min(t_cut1), np.max(t_cut1))
    x2 = np.linspace(np.min(t_cut2), np.max(t_cut2))
    plt.plot(x1, 0.001*gerade(x1, *params1), "k", label="Regression")
    plt.plot(x2, 0.001*gerade(x2, *params2), "k")
    plt.errorbar(t, 0.001*noms(p), xerr=0.2,     yerr=0.001*stds(p),     color='blue', ecolor='red',  markersize=3.5, elinewidth=1, fmt='x', label="Daten")
    plt.xlabel(r"$t [s]$")
    plt.ylabel(r"$p [hPa]$")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.savefig('build/Turbo_Leck_'+data[i]+'.pdf')

    #print("\nBerechnung des Saugvermögens")
    V0  = ufloat(34*10**(-3), 3.4*10**(-3))
    S1 = params1_u[0]*V0/p_g[i]
    S2 = params2_u[0]*V0/p_g[i]
    print(f"p1={(p_cut1[0]+p_cut1[-1])/2}e-3hPa")
    print(f"S1={S1*3600}m³/h")
    print(f"p2={(p_cut2[0]+p_cut2[-1])/2}e-3hPa")
    print(f"S2={S2*3600}m³/h")

for i in [3]:
    print("\nMessung "+data[i])
    t, p1, p2 = np.genfromtxt('data/Turbo_Leck_'+data[i]+'.dat', unpack=True)
    p_m = (p1+p2)/2
    p_sys = np.multiply(p_m,err)
    p_stat = np.sqrt(((p1-p_m)**2+(p2-p_m)**2))
    np.savetxt('build/Turbo_Leck_4_Daten.txt', np.column_stack([t, p_m, p_stat, p_sys]), fmt='%6.5f', delimiter=' & ', header='p, p_stat, p_sys', newline='\\\\\n' )
    p = unp.uarray(p_m, p_sys)

    #Fit
    t_cut1=t[5:cut1[i]]
    p_cut1=p[5:cut1[i]]
    params1,covariance_matrix1 = curve_fit(gerade, t_cut1, noms(p_cut1), sigma=stds(p_cut1), absolute_sigma=True)
    print("Die Parameter der ersten Regression sind:")
    errors1     = np.sqrt(np.diag(covariance_matrix1))
    for name, value, error in zip('ab', params1, errors1):
        params1_u = unp.uarray(params1, errors1)
        print(f'{name} = {value:.8f} ± {error:.8f}')

    t_cut2=t[cut2[i]:cut3[i]]
    p_cut2=p[cut2[i]:cut3[i]]
    params2,covariance_matrix2 = curve_fit(gerade, t_cut2, noms(p_cut2), sigma=stds(p_cut2), absolute_sigma=True)
    print("Die Parameter der zweiten Regression sind:")
    errors2     = np.sqrt(np.diag(covariance_matrix2))
    for name, value, error in zip('ab', params2, errors2):
        params2_u = unp.uarray(params2, errors2)
        print(f'{name} = {value:.8f} ± {error:.8f}')


    #Plot
    plt.figure()
    x1 = np.linspace(np.min(t_cut1), np.max(t_cut1))
    x2 = np.linspace(np.min(t_cut2), np.max(t_cut2))
    plt.plot(x1, 0.001*gerade(x1, *params1), "k", label="Regression")
    plt.plot(x2, 0.001*gerade(x2, *params2), "k")
    plt.errorbar(t, 0.001*noms(p), xerr=0.2,     yerr=0.001*stds(p),    color='blue', ecolor='red',  markersize=3.5, elinewidth=1, fmt='x', label="Daten")
    plt.xlabel(r"$t [s]$")
    plt.ylabel(r"$p [hPa]$")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid()
    plt.savefig('build/Turbo_Leck_'+data[i]+'.pdf')

    #print("\nBerechnung des Saugvermögens")
    V0  = ufloat(34*10**(-3), 3.4*10**(-3))
    S1 = params1_u[0]*V0/p_g[i]
    S2 = params2_u[0]*V0/p_g[i]
    print(f"S1={S1*3600:.3}m³/h")
    print(f"p1={noms((p_cut1[0]+p_cut1[-1])/2):.3}+-{noms((p_cut1[0]-p_cut1[-1])/2):.3}e-3hPa")
    print(f"S2={S2*3600:.3}m³/h")
    print(f"p2={noms((p_cut2[0]+p_cut2[-1])/2):.3}+-{noms((p_cut2[0]-p_cut2[-1])/2):.3}e-3hPa")
