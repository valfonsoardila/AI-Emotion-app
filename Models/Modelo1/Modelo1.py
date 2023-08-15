#Modelo de red neuronal BackPropagation#
import tensorflow as tf
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img


class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') <= 0.0000001:
            print("\nAlcanzado el loss deseado. Deteniendo el entrenamiento.")
            self.model.stop_training = True
            self.model.save("Modelo2/Modelo2.h5")

# cargamos de los datos
x = np.load("Data/data/Train.npy")
y = np.load("Data/data/Train_Labels.npy")

#nomalizacion
x= x/255
# Creamos el modelo 
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(48,40)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(7, activation=tf.nn.softmax)
])

# Compilamos el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# # Tamaño de los lotes de entrenamiento si t_bach==1 recorre patron por patron
t_bach = 1
historial = modelo.fit(x,y,t_bach,epochs=150,callbacks=[LossCallback()])


