from cProfile import label
from os import replace
from turtle import color
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import csv

"""
-Plott termoelektrisk spenningen \epsilon som funskjon av temperatur T_B
-Tilpass en linær kurve til datapunktene vha. minste kvadrater metode 
-Prøv deretter å tilpasse et 2. grads polynom til datapunktene 
"""

# Reading Data
data = pd.read_csv("termo_forsøk_1.txt")
print(data.shape)
print(data.head())

X = data["Temperature (*C) koking til 100"].values
Y = data["Voltage (mV) koking til 100"].values
plt.scatter(X,Y)
plt.show()

# Collecting X and Y
X = data["Temperature (*C) koking til 100"].values
Y = data["Voltage (mV) koking til 100"].values

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values 
n = len(X)

# Using the formula to calculate m and c
numer = 0 
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
m = numer / denom 
c = mean_y - (m * mean_x)

# Printing coefficients 
print("Coefficients")
print(m, c)

# Plotting values and regression line 
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calcualting line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x

# Ploting line 
plt.plot(x, y, color = "red", label="Regresjonslinje")
plt.scatter(X, Y, label="Spredningsplott")
plt.xlim(13,110)
plt.ylim(0.45,4.151)


plt.xlabel("Temperatur ($T_B$)")
plt.ylabel("Termoelektrisk spenning ($\epsilon$)")
plt.legend()
plt.show()
