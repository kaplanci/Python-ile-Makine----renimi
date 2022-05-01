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

veriler = pd.read_csv("eksikveriler.csv")


#veri ön işleme

print (veriler)

# =============================================================================
# boy = veriler[["boy"]]
# print (boy)
# 
# boykilo = veriler [["boy","kilo"]]
# print(boykilo)
# 
# 
# class insan :
#     boy = 180
#     def kosmak(self , b):
#         return b + 10   
# 
# ali = insan ()
# print (ali.boy)
# print (ali.kosmak(90))
# 
# l = {1,2,3,4} # listeler
# =============================================================================

               #eksik veriler ekleme çok önemli
              


yas  = veriler.iloc[:,1:4].values





ulke = veriler.iloc[:,0:1].values
print (ulke)



#encoder işlemi nominal to ordinal
#kategorik verileri nümerik verilere dönüştürme işlemi



from sklearn import preprocessing
le = preprocessing.LabelEncoder()


ulke [:,0]= le.fit_transform(veriler.iloc[:,0])  #öğrenme ve uygualama fonskiyonunu aynı anda çağırdık

print(ulke)

#one hot encoder amacı kolon başlıklarına etiket taşımaktır




ohe = preprocessing.OneHotEncoder()    #oneHotEncoder çok öenmli 3lü sistem

ulke = ohe.fit_transform(ulke).toarray()


print (ulke)


#sonuç olark kategorik değerleri nümerik değerlere dönüştürdük


#numpy dizileri dataframe dizliere dönüştürüldü


sonuc = pd.DataFrame(data = ulke , index = range(22), columns = ["fr","tr","us"])

print(sonuc)

sonuc2 = pd.DataFrame(data = yas , index = range(22), columns = ["boy","kilo","yas"] )

print(sonuc2)


cinsiyet = veriler.iloc[:,-1].values

print(cinsiyet)


sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ["cinsiyet"])

print(sonuc3)


#concat ile dataframleri birleştirdik...



s  = pd.concat([sonuc,sonuc2], axis = 1)
print (s)

s2 = pd.concat ([s,sonuc3],axis = 1)

print(s2)





                #VERİLERİ BÖLME
                
                
 #veriler test ve eğitim için bölündü               
        

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33,random_state=0)

 #bağımlı değişken cinsiyet
 
 #verinin teste ve traine dahil olcağaı dört ayrı kümeye bölmüş olduk x ekseninde ve y ekseninde
 
 
 
 
 
 ##öznitelik ölçekleme farklı dataları birbiirne benzettik aşağıdaki kodla
 
 #standartlaştırma  işlemi
 
 
from sklearn.preprocessing import StandardScaler
  
sc = StandardScaler()
  
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
  
 ##şimdi hazır bir şablon oluşturacağız sonradan kullanılmak üzere
 
 
#yukarısı data processing işlemleri olup sonradan kullanılmak üzere bir genel şablon tanıtımıdır
 
 
 
 
 
 
 















               
               
               


        






























