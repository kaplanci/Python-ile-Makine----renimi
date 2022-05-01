# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:46:42 2022

@author: Batuhan
"""
#kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as pld

 
#veri ön işleme
  #veri yükleme

veriler = pd.read_csv("veriler.csv")


#veri ön işleme

# =============================================================================
# print (veriler)
# 
# boy = veriler[["boy"]]
# print (boy)
# 
# boykilo = veriler [["boy","kilo"]]
# print(boykilo)
# 
# =============================================================================

yas  = veriler.iloc[:,1:4].values

           ##Ülke encoding işlemi


ulke = veriler.iloc[:,0:1].values
print (ulke)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


ulke [:,0]= le.fit_transform(veriler.iloc[:,0])  #

print(ulke)

ohe = preprocessing.OneHotEncoder()    

ulke = ohe.fit_transform(ulke).toarray()


print (ulke)



    ##cinsiyet encoding dönüşümü



c = veriler.iloc[:,-1:].values
print (c)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


c [:,-1]= le.fit_transform(veriler.iloc[:,-1]) 

print(c)


ohe = preprocessing.OneHotEncoder()    #

c = ohe.fit_transform(c).toarray()

print (c)


#sonuç olark kategorik değerleri nümerik değerlere dönüştürdük





sonuc = pd.DataFrame(data = ulke , index = range(22), columns = ["fr","tr","us"])

print(sonuc)

sonuc2 = pd.DataFrame(data = yas , index = range(22), columns = ["boy","kilo","yas"] )

print(sonuc2)


cinsiyet = veriler.iloc[:,-1].values

print(cinsiyet)


sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ["cinsiyet"])

print(sonuc3)


#concat ile dataframleri birleştirdik...



s  = pd.concat([sonuc,sonuc2], axis = 1)
print (s)

s2 = pd.concat ([s,sonuc3],axis = 1)

print(s2)



                
           
        

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33,random_state=0)


 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

##boy tahmini için grafiksel manipülasyonlar

boy= s2.iloc[:,3:4]




sol = s2.iloc[:,:3]



sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag], axis= 1)



x_train, x_test, y_train, y_test = train_test_split(veri,boy,test_size = 0.33,random_state=0)


r2 = LinearRegression()

r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)




                 ## BACKWARD ELİMİNATİON ŞABLON


import statsmodels.api as sm


X = np.append(arr = np.ones((22,1)).astype(int),values = veri, axis = 1)


x_l = veri.iloc[:,[0,1,2,3,4,5]].values

x_l = np.array (x_l,dtype=float)

model = sm.OLS(boy,x_l).fit()

print(model.summary())


#OLS istatiklerinden p value değeri en yüksek olan 4.kolonu atıyoeuz


x_l = veri.iloc[:,[0,1,2,3,5]].values

x_l = np.array (x_l,dtype=float)

model = sm.OLS(boy,x_l).fit()

print(model.summary())


#5i de atttık

x_l = veri.iloc[:,[0,1,2,3]].values

x_l = np.array (x_l,dtype=float)

model = sm.OLS(boy,x_l).fit()

print(model.summary())
































 
 
 
 
 
 
 
 
 
 
 
 















               
               
               


        






























