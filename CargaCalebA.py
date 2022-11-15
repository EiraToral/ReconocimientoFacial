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

#Se definen los datos de entrenamiento y de prueba
#Para fines prácticos se usan 50000 imágenes
num_train = 40000
epochs = 10
num_test = len(df) - num_train
epochs_step = num_train // batch_size
test_step = num_test // batch_size
data_train = labeled_images.take(num_train)
data_test = labeled_images.skip(num_train)

#Se define el modelo de red neuronal
#Para este entrenamiento se utilizan 2 capas convolucioneles con función de activación RELU y un PaxPooling de 2x2
#model = Sequential()
inputs = keras.Input(shape=(192, 192, 3), name='input')
x = tf.keras.layers.Conv2D(10, (3, 3))(inputs)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(10, (3, 3))(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(10, (3, 3))(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
#Se agrega una activavión Dropout y se vectoriza las salidas de la última capa
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
#Se agragan dos capas densas, la última con una salida de 40 neuronas
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(40, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=output) #Se define la entrada y salida
#Se agraga ek rensorboard y un cronómetro para medir el tiempo
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

model.summary()

#Se configura el optimizador, la función de costo y las métricas
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()],
              run_eagerly=True)

#Se entrena el modelo con los datos de entrenamiento y pruebas definidos previamente
model.fit(data_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=data_test,
    callbacks=[tbCallBack])

#Se guarda el modelo en el disco
model.save('Modelo_entrenado_celeba.h5')

###
