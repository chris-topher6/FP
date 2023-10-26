#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TEM00 = pd.read_csv("./data/TEM00.txt", sep="\s+", header=1)
TEM10 = pd.read_csv("./data/TEM10.txt", sep="\s+", header=1)
