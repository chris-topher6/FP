import scipy.constants as const
import numpy as np


#A1 - Lebensdauer
print("A1")
#Konstanten
s   = 10000         #m
m   = 105.7         #MeV
tau = 2.197*10**(-6)#s
E   = 10000         #MeV

beta = np.sqrt(E**2-m**2)/E
v    =beta*const.c

#klassisch
print("klassisch:")
t = s/v
print(f"das Myon benoetigt {t/tau} Lebensdauern")
print(f"die Wahrscheinlichkeit, dass ein Myon ankommt liegt bei {np.exp(-t/tau):.4}\n")

#relativistisch
print("relativistisch:")
t = t*np.sqrt(1-beta**2) #Zeitdilatation
print(f"das Myon benoetigt {t/tau} Lebensdauern")
print(f"die Wahrscheinlichkeit, dass ein Myon ankommt liegt bei {np.exp(-t/tau):.4}\n")

#A2 - Ereignisrate
print("A2")
#Konstanten
V = 50000 #cm^3
ER= 1    # #mu/(cm^2*min) <-Ereignisrate
ER=ER/60 # #mu/(cm^2*s)

r = (V/(2*np.pi))**(1/3) #cm
h = 2*r
A = 2*r*h #cm^2
print(f"{ER*A:.3} Myonen strömen pro Sekunde durch die Detektor-Querfläche\n")

#A3 - Untergrundrate
print('A3')
Ts  =   10**(-6) #Suchzeit
l   =   ER*A*Ts  #Ereignisse im Detektor in Suchzeit
k   =   1        #Ereignisszahl
p   =   l**k/(np.math.factorial(k))*np.exp(-l) #Poissonverteilung p=l^k/k! e^-l
print(f"Wahrscheinlichkeit, dass Muon während Suchzeit eintritt liegt bei {p*100:.2}%\n")