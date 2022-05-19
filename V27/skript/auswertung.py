
import numpy as np
from uncertainties import ufloat
import scipy.constants as c

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
    435*10**-3,     #rot
    318*10**-3,     #blau sigma
    435*10**-3      #blau pi
]
for i in range(N):
    print(f"Messreihe{i+1}:")
    x, y  = np.genfromtxt('data/mess'+str(i+1)+'.dat', unpack=True) #Daten importieren

    #Entsorgung der x=0 Elemente
    indexes=np.ones(10)
    indexes=indexes*9
    n=0
    for k in range(len(x)):             #erstellt Array mit den Indizes der Elemente, die null sind
        if x[k]==0:
            indexes[n]=k
            n=n+1
    indexes = indexes.astype(int)
    x = np.delete(x, indexes)           #Elemente werden gelöscht
    s1  = np.zeros(len(x)-1)
    s2  = np.zeros(int((len(y))/2))
    for j in range(len(s1)):
        s1[j]=x[j+1]-x[j]               #x[1]-x[0], x[2]-x[1], x[3]-x[2] usw.
    for j in range(len(s2)):
        s2[j]=y[2*j+1]-y[2*j]           #x[1]-x[0], x[3]-x[2], x[5]-x[4] usw.
    ms1 = ufloat(np.mean(s1), np.std(s1))
    ms2 = ufloat(np.mean(s2), np.std(s2))
    l=(ms2/ms1)*dlam[i]/2
    a=c.h*c.c*l
    b=lam[i]**2*muB*B[i]
    g=(c.h*c.c*l)/(lam[i]**2*muB*B[i])
    print(f"s1=({ms1:.3})px")
    print(f"s2=({ms2:.3})px")
    print(f" l=({l*10**12:.3})pm")
    print(f" g=({g:.3})\n")