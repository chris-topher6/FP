
import numpy as np
from uncertainties import ufloat
import scipy.constants as c

def abw(lit, exp): #Abweichung in %
    return (np.abs(lit-exp)/lit*100)

#Konstanten
N=3 #Anzahl an .data files
muB=9.2740100783*10**(-24) #Bohrsches Magneton
dlam=[               #m #Dispersionsgebiet
    48.91*10**(-12), #rot
    26.95*10**(-12), #blau
    26.95*10**(-12)  
]
lam=[               #m #Wellenlänge des Lichts
    643.8*10**(-9), #rot
    480.0*10**(-9), #blau
    480.0*10**(-9)  #blau
]
B=[                 #T #Magnetfeld für verschiedene Messreihen
    577.5*10**-3,   #rot
    427.0*10**-3,     #blau sigma
    577.5*10**-3    #blau pi
]
lit=[1,
    1.75,
    0.5
]
for i in range(N):
    print(f"Messreihe{i+1}:")
    x, y  = np.genfromtxt('data/mess'+str(i+1)+'.dat', unpack=True) #Daten importieren
    ms1 = ufloat(np.mean(x), np.std(x))
    ms2 = ufloat(np.mean(y), np.std(y))
    l=(ms2/ms1)*dlam[i]/2
    g=(c.h*c.c*l)/(lam[i]**2*muB*B[i])
    print(f"B={B[i]}T")
    print(f"s1=({ms1:.4})px")
    print(f"s2=({ms2:.4})px")
    print(f"l =({l*10**12:.3})pm")
    print(f"g =({g:.3})")
    print(f"p =({abw(lit[i],g):.2})%\n")