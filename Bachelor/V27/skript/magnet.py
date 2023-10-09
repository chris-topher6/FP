import matplotlib.pyplot as plt
import numpy as np

#Berechnung der Dispersionsgebiete:


def poly3(a,b,c,d,x):
    return a*x**3+b*x**2+c*x+d

I, B = np.genfromtxt('data/magnet.dat', unpack=True)

params, covariance_matrix = np.polyfit(I, B, deg=3, cov=True)
uncertainties = np.sqrt(np.diag(covariance_matrix))

#Ausgeben der Parameter
print("\nRegressionsparameter für vollständige Regression")
errors = np.sqrt(np.diag(covariance_matrix))
for name, value, error in zip('abcd', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

#Plot
x = np.linspace(np.min(I), np.max(I))
x2= np.linspace(np.min(I), 5)
plt.plot(x, poly3(params[0],params[1],params[2],params[3],x), color="darkgreen", label="Regression")
plt.plot(I,B, "x", color="darkblue", label="Daten")
plt.xlabel(r"$I [A]$")
plt.ylabel(r"$B [mT]$")
plt.legend()
plt.tight_layout()
plt.savefig('build/magnet.pdf')

print(f"B(3.0A)={poly3(params[0],params[1],params[2],params[3],3)}")
print(f"B(4.5A)={poly3(params[0],params[1],params[2],params[3],4.5)}")