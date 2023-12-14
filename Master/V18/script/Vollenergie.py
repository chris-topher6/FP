#!/usr/bin/env python3

from uncertainties.umath import *
from uncertainties import ufloat

AKTIVITÄT = ufloat(4130, 60) # Bq, am 1.10.2000
MESSZEIT = 2802 # s
HALBWERTSZEIT_EU = (13.522)*365*24*60*60 # Jahre, in s umgerechnet


def aktivitätsgesetz(t, A0, tau):
    A = A0 * exp(((-log(2)) / tau) * t)
    return A


print(f"Die Aktivität am Messtag")
