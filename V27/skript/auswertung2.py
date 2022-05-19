
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from numpy.linalg import inv

x,y = np.genfromtxt('data/mess3.dat', unpack=True)

s = np.array([x[1]-x[0], x[2]-x[1], x[3]-x[2], x[4]-x[3]])
ms = ufloat(np.mean(s), np.std(s))


s2 = np.array([y[1]-y[0], y[3]-y[2], y[5]-y[4], y[7]-y[6], y[9]-y[8]])

ms2 = ufloat(np.mean(s2), np.std(s2))

lb = 26.95 

l = (ms2/ms)*(lb/2)

l1 = l*10**-12 
h = 6.626*10**(-34)
c = 2.99*10**8
lr = 480*10**-9
mu = 9.274*10**-24
B = 435*10**-3

a=(h*c*l1)
b=(lr**2*mu*B)
g=a/b
print(f"s1=({ms:.3})px")
print(f"s2=({ms2:.3})px")
print(f" l=({l:.3})pm")
print(a)
print(b)
print(f" g=({g:.3})\n")
print(s)
#print(s2)