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

#Dataset es un objeto de TF (Convierte elememtos matriciales, como clumnas o filas, y los convierte en un objeto de tensorflow
files = tf.data.Dataset.from_tensor_slices(df[0]) #Se toma la primera columna (image name) y la transformamos en un objeto de Tensorflow
attributes = tf.data.Dataset.from_tensor_slices(x) #Se convierte la matriz x a objeto TF
data = tf.data.Dataset.zip((files, attributes)) #Se concatena la columna de nombres (files) y la matriz de atributos (x)
path_to_images = 'img_align_celeba/'
#print(data)

def process_file(file_name, attributes):
    #Se cargan las imágenes uniendo la ruta de la carpeta y el nombre de la imagen
    image = tf.io.read_file(path_to_images + file_name)
    #Se descomprime la imagen
    image = tf.image.decode_jpeg(image, channels=3)
    #Se reescala la imagen
    image = tf.image.resize(image, [192, 192])
    #Se normalizan los pixeles
    image /= 255.0
    #Esta función regresa a la imagen en lugar del nombre obtenido de file_name
    #y los atributos de la imagen (sin modificar)
    return image, attributes

batch_size = 50
#Se aplica la función process_file al Dataset concatenado data
labeled_images = data.map(process_file).batch(batch_size) #.map solamente guarda la información y la etiqueta para usarla cuando se requiera
#print(labeled_images)
