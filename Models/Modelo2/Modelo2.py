# import numpy as np
# from scipy.spatial.distance import cdist
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# class RBFNetwork:
#     def __init__(self, n_centroids):
#         self.n_centroids = n_centroids
#         self.centroids = None
#         self.weights = None

#     def fit(self, X, y):
#         # 1. Select centroids
#         self.centroids = X[np.random.choice(X.shape[0], self.n_centroids, replace=False)]

#         # 2. Calculate standard deviation
#         std = np.mean(cdist(X, self.centroids, metric='euclidean'))

#         # 3. Calculate activations
#         activations = np.exp(-cdist(X, self.centroids, metric='euclidean') ** 2 / (2 * std ** 2))

#         # 4. Adjust weights
#         if activations.shape[0] != y.shape[0]:
#         #raise ValueError("Number of examples in activations and y don't match")
#             self.weights = np.linalg.inv(activations.T @ activations) @ activations.T @ y

#     def predict(self, X):
#         std = np.mean(cdist(X, self.centroids, metric='euclidean'))
#         activations = np.exp(-cdist(X, self.centroids, metric='euclidean') ** 2 / (2 * std ** 2))
#         return np.argmax(activations @ self.weights, axis=1)


# # División del conjunto de datos en entrenamiento y prueba
# X_train = np.load("Data/train.npy")
# y_train = np.load("Data/train_Labels.npy")
# X_test = np.load("Data/test.npy")
# y_test = np.load("Data/test_Labels.npy")

# X_test = X_test / 255.0
# X_train = X_train / 255.0

# X_train = np.reshape(X_train,(28708,2304))
# X_test = np.reshape(X_test,(7178,2304))
# print(X_test.shape)

# # Normaliza los valores de los píxeles entre 0 y 1
# # Crea una instancia de la red neuronal de base radial y ajústala a los datos de entrenamiento
# rbf_network = RBFNetwork(n_centroids=100)
# rbf_network.fit(X_train[1:], y_train)

# # Realiza predicciones en los datos de prueba y evalúa el rendimiento
# y_pred = rbf_network.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

#Modelo de red neuronal Convolucional#
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
            self.model.save("Models/Modelo2/Modelo2.h5")

# cargamos de los datos
x = np.load("Data/test.npy")
y = np.load("Data/test_Labels.npy")
print(x.shape)
#nomalizacion
x= x/255
# Creamos el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256,(3,3)),
    tf.keras.layers.MaxPooling2D(2,2),
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
t_bach = 100
historial = modelo.fit(x,y,t_bach,epochs=500,callbacks=[LossCallback()])
modelo.save("Models/Modelo2/Modelo2.h5")
