#!/usr/bin/env python3
import numpy as np
import pandas as pd


def average_frequency_difference(file_path):
    # Daten einlesen
    Frequenz = pd.read_csv(file_path, sep="\s+", header=1)

    # Berechne die Differenzen zwischen aufeinander folgenden Frequenzwerten und speichere sie in einer Liste
    delta_fs = Frequenz["f"].diff().dropna()

    # Berechne die durchschnittliche Differenz
    return delta_fs.mean()


# Liste der Dateipfade
files = [
    "./data/Frequenz1.txt",
    "./data/Frequenz2.txt",
    "./data/Frequenz3.txt",
    "./data/Frequenz4.txt",
]

# Durchschnittliche Differenzen für jede Datei berechnen und ausgeben
for file in files:
    avg_diff = average_frequency_difference(file)
    print(f"The average frequency difference for {file} is {avg_diff:.2f} MHz")
