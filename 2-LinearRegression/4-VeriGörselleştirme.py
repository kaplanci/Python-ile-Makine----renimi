# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:46:42 2022

@author: Batuhan
"""
#kütüphaneleri yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
#veri ön işleme
  #veri yükleme

veriler = pd.read_csv("satislar.csv")


#veri ön işleme

print (veriler)


aylar = veriler[["Aylar"]]
print (aylar)
 
 
satislar = veriler [["Satislar"]]
print(satislar)
 
satislar2  = veriler.iloc[:,0:1].values
print(satislar2)




                #VERİLERİ BÖLME
                
                
 #veriler test ve eğitim için bölündü               
        

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size = 0.33,random_state=0)


 
from sklearn.preprocessing import StandardScaler

'''
sc = StandardScaler()
  
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)

'''


from sklearn.linear_model import LinearRegression

lr =LinearRegression()

lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)


x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.plot(x_train,y_train)
plt.plot(x_test , lr.predict(x_test))


# görselleştirme muahbbeti

plt.title("aylaraGöre satış")

plt.xlabel("aylar")
plt.ylabel("satışlar")


















 
 
 
 
 
 
 















               
               
               


        






























