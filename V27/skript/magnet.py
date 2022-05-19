import matplotlib.pyplot as plt
import numpy as np

I, B = np.genfromtxt('data/magnet.dat', unpack=True)

params, covariance_matrix = np.polyfit(I, B, deg=1, cov=True)
uncertainties = np.sqrt(np.diag(covariance_matrix))

#Ausgeben der Parameter
print("\nRegressionsparameter für Platau für justage 15")
errors = np.sqrt(np.diag(covariance_matrix))
for name, value, error in zip('ab', params, errors):
    print(f'{name} = {value:.8f} ± {error:.8f}')

#Plot
x = np.linspace(np.min(I), np.max(I))
plt.plot(x,params[0]*x+params[1], color="darkgreen", label="Regression")
plt.plot(I,B, "x", color="darkblue", label="Daten")
plt.xlabel(r"$I [A]$")
plt.ylabel(r"$B [mT]$")
plt.legend()
plt.tight_layout()
plt.savefig('build/magnet.pdf')