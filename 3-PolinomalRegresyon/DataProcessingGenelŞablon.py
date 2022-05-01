# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:46:42 2022

@author: Batuhan
"""
#kütüphaneleri yükleme
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
           
        
x = veriler.iloc[:,1:2]  #aldığımız kolonlar regresyona sokcağımız columns
y = veriler.iloc[:,2:]

X=x.values
Y=y.values


# =============================================================================
# =============================================================================
# #    LİNEER REGRESYON MODELİ
# =============================================================================
# =============================================================================


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X,Y)

plt.scatter(X,Y, color = "red")

plt.plot(X,lin_reg.predict(x), color = "blue")
plt.show()


# =============================================================================
# =============================================================================
# # POLİNOMAL REGRESYON
# =============================================================================
# =============================================================================


from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

x_poly = poly_reg.fit_transform(X)  #ikinci dereceye kadar x değerlerini polinomallaştırdık

print(x_poly)

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly,Y)


plt.scatter(X,Y,color = "red")

plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = "blue")  

#predict ederken polinomal dönüşümünü yaptık yukarda

plt.show()




from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(X)  #dördüncü dereceye kadar x değerlerini polinomallaştırdık

print(x_poly)

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly,Y)


plt.scatter(X,Y,color = "red")

plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = "blue")  

#predict ederken polinomal dönüşümünü yaptık yukarda

plt.show()


#tahmini maaşlar
   
    #lineer tahmin

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg.predict([[3.3]]))

    #polinomal tahmin

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


































 
 
 
 
 
 
 
 















               
               
               


        






























