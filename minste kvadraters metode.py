from cProfile import label
from email import header
from turtle import color
from attr import s
from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv("termo_forsøk_2.txt")
print(data.head())

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
lin = LinearRegression()

lin.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)



plt.scatter(X, y, color = "blue", marker=".", s = 10)
plt.plot(X, lin.predict(X), color = "black", label = "Linear reg", linewidth = 2)
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = "red", label = "Least Squares Reg", linewidth = 2)
plt.xlabel("Temperatur")
plt.ylabel("Voltge")
plt.legend()
plt.show()




