
#Modelo de red neuronal Convolucional Con aumento de datos#
import tensorflow as tf
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img

# observamos que el valor de perdida se muy bajo
class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') <= 0.0001:
            print("\nAlcanzado el loss deseado. Deteniendo el entrenamiento.")
            self.model.stop_training = True
            self.model.save("Models/Modelo3/Modelo3.h5")

# cargamos de los datos
x = np.load("Data/test.npy")
y = np.load("Data/test_Labels.npy")
print(x.shape)
#nomalizacion
x= x/255

#aumento de datos
ImagDg=tf.keras.preprocessing.image.ImageDataGenerator
rango_rotacion = 30
mov_ancho = 0.25
mov_alto = 0.25
#rango_inclinacion=15 #No uso este de momento pero si quieres puedes probar usandolo!
rango_acercamiento=[0.5,1.5]
datagen = ImagDg(
    rotation_range = rango_rotacion,
    width_shift_range = mov_ancho,
    height_shift_range = mov_alto,
    zoom_range=rango_acercamiento,
    #shear_range=rango_inclinacion #No uso este de momento pero si quieres puedes probar usandolo!
)
x_expand = x.reshape(x.shape[0], 48, 48, 1)
print(x_expand.shape)
datagen.fit(x_expand)

# Creamos el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(7, activation=tf.nn.softmax)
])

# Compilamos el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# # Tamaño de los lotes de entrenamiento si t_bach==1 recorre patron por patron
t_bach = 120
data_gen_entrenamiento = datagen.flow(x_expand,y, batch_size=t_bach)
historial = modelo.fit(data_gen_entrenamiento,batch_size=t_bach,epochs=500,callbacks=[LossCallback()],steps_per_epoch=int(np.ceil(7542 / float(t_bach))))
modelo.save("Models/Modelo3/Modelo3.h5")
"""_summary_
red: backpropagation
funcion de activacion: relu,softmax
Algorito de entretamiento: Adam
Tipo de red: Multicapa
Numero de capas: 6
Topologia
Input
numero de neuronas:10000
funcion de activacion:relu
Oculta1
numero de neuronas:100
funcion de activacion:relu
Oculta2
numero de neuronas:100
funcion de activacion:relu
Oculta3
numero de neuronas:100
funcion de activacion:relu
Dropuout
rata: 0.5
Oculta4
numero de neuronas:50
funcion de activacion:relu
Output
numero de neuronas:4
funcion de activacion:softmax

Learning rate: 0.0001
erro Maximo Permitido: 0.003
Presicion: 100%

"""

