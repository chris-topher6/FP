import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

# functions
def transform_data(data): # Die Funktion brauchen wir, weil wir die Messung von 10 bis 0 grad und nicht von 0 bis 10 grad gemacht haben
    differences = -np.diff(data)  # Differenzen berechnen und umkehren
    cumulative_sum = np.cumsum(differences, axis=0)
#    cumulative_sum = np.insert(cumulative_sum, 0, 0)  # 0 am Anfang einfügen
    return cumulative_sum

def Maxima(theta, n): # Function to calculate the number of maxima
    return 2*T/lamda * (n-1)/n *alpha_0*theta

def grad_to_rad(theta):
    return theta*np.pi/180 

def rad_to_grad(theta):
    return theta*180/np.pi

# constants
T = 1e-3 # Thickness of the glass pane in meters
alpha_0 = grad_to_rad(10) # Initial angle in radians
lamda = 632.99e-9 

# Reading data from the file and unpacking it into variables
theta, M1, M2, M3, M4, M5, M6, M7, M8, M9 = np.genfromtxt("data/glass.dat", unpack = True)

theta = grad_to_rad(theta)
theta = theta[1:]

# Ab hier transformieren wir die Daten so, als wenn wir von 0 bis 10 und nicht von 10 bis 0 gemessen haben. Die Physik steckt ja eh in der Differenz zwischen den Werten, von daher ist die Reihenfolge egal
transformed_M = []

for data in [M1, M2, M3, M4, M5, M6, M7, M8, M9]: # Transformation für jede Messreihe durchführen
    transformed_M.append(transform_data(data))

transformed_M = np.array(transformed_M) # Umwandeln der Liste in ein NumPy-Array für eine einfachere Handhabung

print(f"{transformed_M}") # Das ist auch das, was als Messdaten ins Protokoll kommt, es wurden ja nur triviale Trafos gemacht

# Berechnung des Mittelwerts für jede Spalte in der Matrix 'transformed_M'
M_mean = np.mean(transformed_M, axis=0)
M_std = np.std(transformed_M, axis=0)

# Printing the mean and standard deviation of the measurements
print("--------------------------------------------------")
for i in range(len(M_mean)):
    print(f"{M_mean[i]:.4f} +- {M_std[i]:.4f}")
print("--------------------------------------------------")

# Curve fitting to determine the refractive index
params1, pcov1 = op.curve_fit(Maxima, theta, M_mean, sigma = M_std)
err1 = np.sqrt(np.diag(pcov1))

# Printing the results
print("--------------------------------------------------")
print("Refractive Index Glass:")
print(f"n = {params1[0]:.4f} +- {err1[0]:.4f}")
print("--------------------------------------------------")

# Plot
x = np.linspace(0, 10, 100)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Plotting the data and Fit
ax1.errorbar(rad_to_grad(theta), M_mean, yerr=M_std, fmt='o', markersize=5, color='black', ecolor='dimgray', elinewidth=2, capsize=2, label="Data")
ax1.plot(x, Maxima(grad_to_rad(x), *params1), color='dodgerblue', linestyle='-', linewidth=2, label="Regression")

# Setting labels and limits for the main plot
ax1.set_ylabel(r"$M$", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# Calculating residuals for the pull plot
pulls = (M_mean - Maxima(theta, *params1)) / M_std
ax2.plot(theta*180/np.pi, pulls, 'o', markersize=5, color='red')
ax2.plot(rad_to_grad(theta), pulls, 'o', markersize=5, color='red')

ax2.axhline(0, color='black', linestyle='--', linewidth=1)  # Zero line for reference
ax2.set_ylim(-1, 1)

# Setting labels and limits for the pull plot
ax2.set_ylabel("Pulls/$\sigma$")
ax2.set_xlabel(r"$\theta / °$", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Verschönern und speichern
plt.tight_layout()
fig.patch.set_facecolor('whitesmoke')
plt.savefig("build/n_glass.pdf", bbox_inches='tight')
plt.close()