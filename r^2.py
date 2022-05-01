# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:46:42 2022

@author: Batuhan
"""
# kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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

print("Lineer r2 değeri")
print(r2_score(Y, lin_reg.predict(X)))



# polinomal tahmin

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print("Polynomial r2 değeri")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


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
print(svr_reg.predict([[6.6]]))


print(" SVR r2 değeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


# =============================================================================
# =============================================================================
# # # karar ağacı ile prediciton ve görselleşirtme
# =============================================================================
# =============================================================================


from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)

plt.scatter(X, Y, color="red")
plt.plot(X, r_dt.predict(X), color="blue")


plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
print("Decision Tree r2 değeri")
print(r2_score(Y, r_dt.predict(X)))



# =============================================================================
# =============================================================================
# # 
# # #RASSAL AĞAÇ REGÜLASYONU
# =============================================================================
# =============================================================================


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)  

'''#estimators kaç tane karar aracağını çizaceğimizin sayısı'''

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.5]]))

plt.scatter(X,Y,color = "red")
plt.plot(X,rf_reg.predict(X),color ="blue")
plt.show()

# =============================================================================
# =============================================================================
# #  R SQUARE HESAPLAMA
# =============================================================================
# =============================================================================
  
from sklearn.metrics import r2_score
print("random forest r2 değeri")
print(r2_score(Y,rf_reg.predict(X)))


# =============================================================================
# =============================================================================
# # OZET R2 DEGERLERİ
# =============================================================================
# =============================================================================

print("----------------------------------")
print("Lineer r2 değeri")
print(r2_score(Y, lin_reg.predict(X)))


print("Polynomial r2 değeri")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


print(" SVR r2 değeri")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print("Decision Tree r2 değeri")
print(r2_score(Y, r_dt.predict(X)))

print("random forest r2 değeri")
print(r2_score(Y,rf_reg.predict(X)))











































