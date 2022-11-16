from os import listdir
from os.path import isfile, isdir, join
import numpy
import datetime
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
import re

ih, iw = 192, 192 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales
#train_dir = 'data/minitrain' #directorio de entrenamiento
#test_dir = 'data/minitest' #directorio de prueba
train_dir = 'C:/Users/tomas/PycharmProjects/ReconocimientoFacial/train/' #directorio de entrenamiento
test_dir = 'C:/Users/tomas/PycharmProjects/ReconocimientoFacial/test/'  #directorio de prueba

num_class = 2 #cuantas clases
epochs = 15 #cuantas veces entrenar. En cada epoch hace una mejora en los parametros
batch_size = 50 #batch para hacer cada entrenamiento. Lee 50 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria
num_train = 1600 #numero de imagenes en train
num_test = 400 #numero de imagenes en test

epoch_steps = num_train // batch_size
test_steps = num_test // batch_size

#se configuran las características de las imágenes de entrenamiento y prueba
gentrain = ImageDataGenerator(rescale=1. / 255.) #indica que reescale cada canal con valor entre 0 y 1.
train = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')
gentest = ImageDataGenerator(rescale=1. / 255)
test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')

#Se carga modelo entrenado con las impagenes de celebA
modelo_cargado = tf.keras.models.load_model('Modelo_entrenado_celeba.h5')

#Se definen las capas del nuevo modelo
model = keras.models.Sequential()
#Se agrega el modelo cargado
model.add(modelo_cargado)
#Se coloca una capa densa y una sola neurona de salida
model.add(Dense(20, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
#antes de entrenar congelamos los parámetros del modelo cargado
for layer in model.layers[:1]:
    layer.trainable = False

#Se coloca un contador y se manda a llamar al tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
model.summary()

#Se utiliza un optimizador RMS prop, una función Crossentropy para la función de costo
#y una metrica de efectividad binaria
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()],
              run_eagerly=True)

#Se fuarda el modelo en el disco
model.fit_generator(
                train,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test,
                validation_steps=test_steps,
                callbacks=[tbCallBack]
                )

model.save('ModeloRF')

