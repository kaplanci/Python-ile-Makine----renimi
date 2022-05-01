# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:46:42 2022

@author: Batuhan
"""
# kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# =============================================================================
# # #veri ön işleme
# #   #veri yükleme
# =============================================================================
# =============================================================================

veriler = pd.read_csv("maaslar.csv")
print(veriler)


x = veriler.iloc[:, 1:2]  # aldığımız kolonlar regresyona sokcağımız columns
y = veriler.iloc[:, 2:]

X = x.values
Y = y.values


# =============================================================================
# =============================================================================
# #    LİNEER REGRESYON MODELİ
# =============================================================================
# =============================================================================


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, Y)


# =============================================================================
# =============================================================================
# # POLİNOMAL REGRESYON
# =============================================================================
# =============================================================================


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

x_poly = poly_reg.fit_transform(
    X
)  # ikinci dereceye kadar x değerlerini polinomallaştırdık


lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly, Y)


poly_reg3 = PolynomialFeatures(degree=4)

x_poly3 = poly_reg3.fit_transform(
    X
)  # dördüncü dereceye kadar x değerlerini polinomallaştırdık

lin_reg3 = LinearRegression()

lin_reg3.fit(x_poly3, Y)


# =============================================================================
# =============================================================================
# # GÖRSELLEŞTİRME
# =============================================================================
#
# =============================================================================

plt.scatter(X, Y, color="red")

plt.plot(X, lin_reg.predict(x), color="blue")
plt.show()


plt.scatter(X, Y, color="red")

plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")

# predict ederken polinomal dönüşümünü yaptık yukarda

plt.show()


plt.scatter(X, Y, color="red")

plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color="blue")

# predict ederken polinomal dönüşümünü yaptık yukarda

plt.show()


# tahmini maaşlar

# lineer tahmin

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg.predict([[3.3]]))

# polinomal tahmin

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


# verilerin ölçeklenmesi


from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()

x_olcekli = sc1.fit_transform(X)


sc2 = StandardScaler()

y_olcekli = sc2.fit_transform(Y)

# =============================================================================
# =============================================================================
# # #SVR Regressinon
# =============================================================================
# =============================================================================


from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")  # radial basis function

svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color="red")

plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color="blue")


plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6]]))


# karar ağacı ile prediciton ve görselleşirtme


from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)

plt.scatter(X, Y, color="red")
plt.plot(X, r_dt.predict(X), color="blue")


print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))



















