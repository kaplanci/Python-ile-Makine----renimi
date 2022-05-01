# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:46:42 2022

@author: Batuhan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pld
 
#veri yükleme

veriler = pd.read_csv("veriler.csv")


#veri ön işleme

print (veriler)

boy = veriler[["boy"]]
print (boy)

boykilo = veriler [["boy","kilo"]]
print(boykilo)


class insan :
    boy = 180
    def kosmak(self , b):
        return b + 10   

ali = insan ()
print (ali.boy)
print (ali.kosmak(90))

l = {1,2,3,4} # listeler

#eksik veriler 


        






























