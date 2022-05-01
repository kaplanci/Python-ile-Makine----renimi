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

print (veriler)

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


                #VERİLERİ BÖLME
                
                
 #veriler test ve eğitim için bölündü               
        

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33,random_state=0)

 
 
 
 
 
 ##öznitelik ölçekleme farklı dataları birbiirne benzettik aşağıdaki kodla
 
 #standartlaştırma  işlemi
 
 
from sklearn.preprocessing import StandardScaler
  
sc = StandardScaler()
  
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
  
 
    
 
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(x_train , y_train)  
 
y_pred = logr.predict(x_test)
print(y_pred) 

print(y_test)

# =============================================================================
# =============================================================================
# # 
# #     KARMAŞIKLIK MATRİSİ
# =============================================================================
# 
# =============================================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test , y_pred)

print("Logisitic regression")
print("---------------------")

print(cm)

# outlierlar çıkınca karmaşıklık matrisinden daha güzel bir sonuç aldık




from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1 , metric = "minkowski")

# komşuyu arttırmak her zaman başarılı çalışmaz

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
print("KNN algoritması")
print("---------------------")


print(cm)



























               
               
               


        






























