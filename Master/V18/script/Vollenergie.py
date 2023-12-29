#!/usr/bin/env python3

import numpy as np
import pandas as pd
from uncertainties.umath import *
from uncertainties import ufloat
from datetime import datetime

start_aktivität = ufloat(4130, 60)  # Bq, am 1.10.2000

start_date = datetime(2000, 10, 1)
end_date = datetime(2023, 11, 20)

# Differenz zwischen den Daten
time_difference = end_date - start_date

# Umwandlung der Zeitdifferenz in Sekunden
time_difference_in_seconds = time_difference.total_seconds()

halbwertszeit_eu = (13.522) * 365.25 * 24 * 60 * 60  # Jahre, in s umgerechnet


def aktivitätsgesetz(t, A0, tau):
    """Formel für die Aktivität einer Probe"""
    A = A0 * exp(((-log(2)) / tau) * t)
    return A


end_aktivität = aktivitätsgesetz(
    time_difference_in_seconds, start_aktivität, halbwertszeit_eu
)
a = 8.91  # cm
r = 2.25  # cm
omega_4pi = 1 / 2 * (1 - a / (np.sqrt(a**2 + r**2)))
print(f"Die Aktivität am Messtag betrug {end_aktivität}")
print(f"Der Raumwinkel Omega/4pi beträgt {omega_4pi:.5f}")

peaks = pd.read_csv("./build/peaks.csv")


def fedp(omega, N, A, W, t):
    """Formel für die Vollenergienachweiswahrscheinlichkeit"""
    Q = (4 * np.pi) / omega * N / (A * W * t)
    return Q


# peaks["fedp"] = fedp((omega_4pi*4*np.pi),)
