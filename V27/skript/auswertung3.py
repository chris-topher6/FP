import numpy as np
from uncertainties import ufloat
import scipy.constants as c

def abw(lit, exp): #Abweichung in %
    return (np.abs(lit-exp)/lit*100)

#Konstanten
muB=9.2740100783*10**(-24) #Bohrsches Magneton
dlam = 26.95*10**(-12)     #Dispersionsgebiet
lam  = 480.0*10**(-9)      #Wellenlänge Licht
B=[ 577.5*10**-3,          #Magnetfeldstärke
    302.9*10**-3
]
lit = 1.75                 #Literaturwert für Landé Faktor

x1, y1  = np.genfromtxt('data/mess2.dat', unpack=True) #Daten importieren
ms1 = ufloat(np.mean(x1), np.std(x1)) #mittelwerte der Daten
ms2 = ufloat(np.mean(y1), np.std(y1))
l=(ms2/ms1)*dlam/2               #Breite der Aufspaltung
g1=(c.h*c.c*l)/(lam**2*muB*B[0])  #Landé Faktor: g=chl/(2*muB*lam B)

x2, y2  = np.genfromtxt('data/mess4.dat', unpack=True) #Daten importieren
ms1 = ufloat(np.mean(x2), np.std(x2)) #mittelwerte der Daten
ms2 = ufloat(np.mean(y2), np.std(y2))
l=(ms2/ms1)*dlam/2               #Breite der Aufspaltung
g2=(c.h*c.c*l)/(lam**2*muB*B[1])  #Landé Faktor: g=chl/(2*muB*lam B)

g=(g1*len(x1)+g2*len(x2))/(len(x1)+len(x2))
print(len(x1))
print(len(x2))
print(f"g =({g:.3})")
print(f"p =({abw(lit,g):.3})%\n")