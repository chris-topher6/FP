import numpy as np
import matplotlib.pyplot as plt

def stability(r1, r2, L):
    g1 = 1-L/r1
    g2 = 1-L/r2 
    return g1*g2 

r = [1, 1.4, 10**9]
L = np.linspace(0,2.3,100)

plt.plot(L, stability(r[1], r[1], L), label='c 1400, c 1400')
plt.plot(L, stability(r[1], r[2], L), label='c 1400, f')
plt.grid()
plt.legend()
plt.tight_layout()
#plt.ylim(0, 1) 
plt.savefig("build/stability_theory.pdf")