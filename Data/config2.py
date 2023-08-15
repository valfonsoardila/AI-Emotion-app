import os
import cv2
import numpy as np
import random as rn
import shutil
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import image as img
import tensorflow as tf


"""
            ****IMPORTANTE****
    1. las imagenes se trabjan en 48x48
    2. el banco de datos de entrenamiento tiene 28709 patrones
    3. el banco de datos cuenta con 7 gestos:
        ---Ingles  | Español----
        - anegry   | enojado     
        - disgust  | asco        
        - fear     | miedo       
        - happy    | feliz       
        - neutral  | neutral     
        - sad      | triste      
        - surprise | soprendido 
"""

dirD = 'datafaceDepurate/train'
direccion = 'dataface/train'
def Recoleccion(direccion):
    
    name = str.split(direccion,"/")
    lista = os.listdir(direccion)
    datos =[]
    label =[]
    for i in range(len(lista)):
        contador = 0
        dir2 = direccion+'/'+lista[i]
        lista2 = os.listdir(dir2)
        os.makedirs(direccion+'/'+lista[i])
        for j in range(len(lista2)):
            contador+=1
            dir3 = dir2+'/'+lista2[j]
            imag = img.imread(dir3)
            datos.append(imag)
            label.append(lista[i]+f"{j+1}")
        print(lista[i]+" Procesada")
    
    np.save(f"{name[1]}.npy",np.asarray(datos))
    np.save(f"{name[1]}_Labels.npy",np.asarray(label))

def Recoleccion2(direccion):
    name = str.split(direccion,"/")
    lista = os.listdir(direccion)
    datos =[]
    ldatos =[]
    label =[]
    mat = []
    for i in range(len(lista)):
        contador = 0
        dir2 = direccion+'/'+lista[i]
        lista2 = os.listdir(dir2)
        for j in range(len(lista2)):
            contador+=1
            dir3 = dir2+'/'+lista2[j]
            imag = img.imread(dir3)
            datos.append(imag)
            ldatos.append(i)
            
        label.append([lista[i],i])
        print(lista[i]+" Procesada")

    # tx = tf.convert_to_tensor(np.array(datos))
    # ty = tf.convert_to_tensor(np.array(ldatos))
    # dset = tf.data.Dataset.from_tensor_slices((tx, ty))
    # filename = 'dataset.tfrecord'
    # tf.data.experimental.save(dset,filename)
    # writer.write(dset)
    np.save(f"{name[1]}.npy",np.asarray(datos))
    np.save(f"{name[1]}_Labels.npy",np.asarray(ldatos))
    np.save("Labels.npy",np.asarray(label))

def Recoleccion3(direccion):
    name = str.split(direccion,"/")
    lista = os.listdir(direccion)
    x =[]
    y =[]
    label =[]
    datos = []
    for i in range(len(lista)):
        contador = 0
        dir2 = direccion+'/'+lista[i]
        lista2 = os.listdir(dir2)
        print("Procesando categoria "+lista[i])
        for j in tqdm(range(len(lista2))):
            contador+=1
            dir3 = dir2+'/'+lista2[j]
            imag = img.imread(dir3)
            datos.append([imag,i])
        label.append([lista[i],i])
        print("Categoria "+lista[i]+" Procesada")
    rn.shuffle(datos)
    print("Combinando datos")
    for i in tqdm(range(len(datos))):
        dato = datos[i]
        x.append(dato[0])
        y.append(dato[1])
    np.save(f"{name[1]}.npy",np.asarray(x))
    np.save(f"{name[1]}_Labels.npy",np.asarray(y))
    np.save("Labels.npy",np.asarray(label))
#Recoleccion3(direccion)


def validarcarpeta(carpeta):
    if os.path.exists(carpeta):
        pass
    else:
        os.makedirs(carpeta)

def depuracion(direccion,oupt):
    validarcarpeta(oupt)
    lista = os.listdir(direccion)
    for i in range(len(lista)):
        dir2 = direccion+'/'+lista[i]
        oupt2 = oupt+'/'+lista[i]
        validarcarpeta(oupt2)
        lista2 = os.listdir(dir2)
        print(lista[i]+" en proceso")
        for j in tqdm(range(len(lista2))):
            dir3 = dir2+'/'+lista2[j]
            saveDir = oupt2+'/'+lista2[j]
            img =cv2.imread(dir3)
            cv2.imshow(f"{lista2[j]}",img)
            t=cv2.waitKey(0)
            if t==32:
                shutil.copyfile(dir3,saveDir)
                cv2.destroyAllWindows()
            elif t==27:
                cv2.destroyAllWindows()
        print(lista[i]+" Procesada")

depuracion(direccion,dirD)


# t1 = tf.constant(np.array([[[2,2],[2,2]]],dtype=np.float64),dtype=tf.float64,)
# t2= tf.constant(np.array([2],dtype=np.int8), dtype=tf.int8)
# lista= []
# lista.append(t1)
# lista.append(t2)

# # t3 = tf.constant([[t1],[t2]])

#
#x = np.load("test.npy")
#
#y = np.load("train_Labels.npy")

# print(f"x = {x.shape}")
#print(y)

# # x = np.zeros((7178, 48, 48))  # Matriz x con dimensiones (7178, 48, 48)
# # y = np.zeros((7178,))  # Matriz y con dimensiones (7178,)

# # Expande la matriz y para que tenga las mismas dimensiones que x
# y_expanded = np.expand_dims(y, axis=(1, 2))

# # Concatena las matrices x e y_expanded a lo largo del eje 0 (unión de matrices)
# z = np.array([[[2,2],[2,2]],[3]])


# print(z.shape)  # Salida: (7179, 48, 48)