# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:46:42 2022

@author: Batuhan
"""
# =============================================================================
# =============================================================================
# # #kütüphaneleri yükleme
# =============================================================================
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as pld


# =============================================================================
# =============================================================================
# #  
# # #veri ön işleme
# #   #veri yükleme
# =============================================================================
# 
# =============================================================================
veriler = pd.read_csv("odev_tenis.csv")



# =============================================================================
# =============================================================================
# # 
# # 
# #            # encoding işlemi
# =============================================================================
# =============================================================================


from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
#tüm veriler için label encodind yapma kısayolu
c = veriler2.iloc[:,:1]



from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()   
print (c)

#tablo yapma düzenlemeleri

havadurumu =  pd.DataFrame (data = c ,index= range(14), columns = ['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis = 1)


#veri ön işleme işi bitti şimdi model kurmaya geçiyoruz


# =============================================================================
# =============================================================================
# # VERİLERİN EĞİTİM VE TEST İÇİN BÖLÜNMESİ
# =============================================================================
# 
# =============================================================================


                
           

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size = 0.33,random_state=0)


 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print (y_pred)




# =============================================================================
# =============================================================================
# #                  ## BACKWARD ELİMİNATİON ŞABLON 
# =============================================================================
# =============================================================================


import statsmodels.api as sm


X = np.append(arr = np.ones((14,1)).astype(int),values = sonveriler.iloc[:,:-1], axis = 1)


x_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values  #bağımlı değişken

x_l = np.array (x_l,dtype=float)

model = sm.OLS(sonveriler.iloc[:,-1:],x_l).fit()  #bağımsız değişken

print(model.summary())







sonveriler = sonveriler.iloc[:,1:]  #ilk kolonun p value yüksek attık


import statsmodels.api as sm


X = np.append(arr = np.ones((14,1)).astype(int),values = sonveriler.iloc[:,:-1], axis = 1)


x_l = sonveriler.iloc[:,[0,1,2,3,4]].values  #bağımlı değişken

x_l = np.array (x_l,dtype=float)

model = sm.OLS(sonveriler.iloc[:,-1:],x_l).fit()  #bağımsız değişken

print(model.summary())


x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#sistem iyileşti







 
 
 
 
 
 
 















               
               
               


        






























