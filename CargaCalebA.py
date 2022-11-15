import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime

np.set_printoptions(precision=4)

#Se carga el recuadro gráfico data framme con los espacios corregidos
df = pd.read_csv('../../Downloads/attr_celeba_prepared.txt', sep =' ', header = None)
#print(df) #para observar el dataset corregido
#print(df)
#print (df[0].head) #Imprime los primeros elementos (.head) de la primera columna (df[0])

#A contunuación se configuran los valores del dataset df para que sus valores sean únicamente 0 y 1
df.replace(to_replace=-1, value=0, inplace=True) #Reemplaza los valores "-1" por 0
df.replace(to_replace='0', value=0, inplace=True) #Reemplaza los caracteres '0' por 0
df.replace(to_replace='-1', value=0, inplace=True) #Reemplaza los carateres '1' por 1
df.replace(to_replace='1', value=1, inplace=True) #Reemplaza los caracteres '1' por 1
#Se toman todas las filas y a partir de la segunda columna del dataframe y se transforma a una matriz numpy
x = np.asarray(df.iloc[:, 1:]).astype('int64')
#print(x)