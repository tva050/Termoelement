from cProfile import label
from operator import mod
from turtle import color
from attr import s
import matplotlib.pyplot as plt
import pandas as pd 
import statsmodels.api as sm 

plt.style.use("ggplot")

"""
-Plott termoelektrisk spenningen \epsilon som funskjon av temperatur T_B
-Tilpass en linær kurve til datapunktene vha. minste kvadrater metode 
-Prøv deretter å tilpasse et 2. grads polynom til datapunktene 
"""

data = pd.read_csv("termo_forsøk_1.txt")
print(data.head())

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(X,y)

print("-------------------------------")
print("Coefficient:",lin.coef_)
print("Intercept:", lin.intercept_)
print("-------------------------------")

def ssrlin():
    x = data["Temperature (*C) koking til 100"]
    y = data[["Voltage (mV) koking til 100"]]
    
    x = sm.add_constant(x)
    model = sm.OLS(y,x).fit()
    return print(model.ssr, model.summary())
print(ssrlin())

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)

lin2 = LinearRegression().fit(X_poly,y)
"""lin2.fit(X_poly, y)"""

polynomial_features= PolynomialFeatures(degree=2)
xp = polynomial_features.fit_transform(X)
model4 = sm.OLS(y, xp).fit()
print(f'Summen av residualene til polynomial: {model4.ssr}')
print("-------------------------------")


plt.scatter(X, y, color = "black", marker=".", s = 20, alpha = 0.5)
plt.plot(X, lin.predict(X), color = "blue", label = "Minste kvadrater metode", linewidth = 2.5, alpha = 1)
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = "red", label = "2. grads polynom", linewidth = 2.5, alpha = 1)
plt.xlabel("Temperatur ($\degree C$)", fontname = "serif")
plt.ylabel("Termoelektrisk spenning (mV)", fontname = "serif")
plt.rcParams["font.family"] = "serif"
plt.legend()
plt.show()

plt.scatter(X, y, color = "black", marker=".", s = 20, alpha = 0.5)
plt.xlabel("Temperatur ($\degree C$)", fontname = "serif")
plt.ylabel("Termoelektrisk spenning (mV)", fontname = "serif")
plt.legend()
plt.show()


