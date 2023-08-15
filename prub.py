import cv2
import math
import numpy as np
import mediapipe as mp
import tensorflow as tf

colors=[
    (0, 0, 255), # Azul (Blue)
    (128, 0, 128), # Púrpura (Purple)
    (0, 255, 255), # Cian (Cyan)
    (0, 255, 0), # Verde (Green)
    (128, 128, 128), # Gris (Gray)
    (255, 0, 0), # Rojo (Red)
    (255, 255, 0), # Amarillo (Yellow)
]

def seleccionar(imagen,clase,y1,y2,x3,x4,plus):
        if clase == 0:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),colors[clase],2)
            cv2.putText(imagen,"Persona Enojada",(x3-plus,y1-plus-5),1,1.5,colors[clase],2)
        elif clase == 1:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),colors[clase],2)
            cv2.putText(imagen,"Persona Asqueado",(x3-plus,y1-plus-5),1,1.5,colors[clase],2)
        elif clase == 2:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),colors[clase],2)
            cv2.putText(imagen,"Persona Asustada",(x3-plus,y1-plus-5),1,1.5,colors[clase],2)
        elif clase == 3:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),colors[clase],2)
            cv2.putText(imagen,"Persona Feliz",(x3-plus,y1-plus-5),1,1.5,colors[clase],2)
        elif clase == 4:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),colors[clase],2)
            cv2.putText(imagen,"Persona Neutral",(x3-plus,y1-plus-5),1,1.5,colors[clase],2)

        elif clase == 5:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),colors[clase],2)
            cv2.putText(imagen,"Persona Triste",(x3-plus,y1-plus-5),1,1.5,colors[clase],2)
            
        elif clase == 6:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),colors[clase],2)
            cv2.putText(imagen,"Persona Sorprendida",(x3-plus,y1-plus-5),1,1.5,colors[clase],2)
            
        else:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),(117,29,83),2)
            cv2.putText(imagen,"Persona detectada",(x3-plus,y1-plus-5),1,1.5,(117,29,83),2)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
index_list =[10,152,234,454]
cap = cv2.VideoCapture(1)
modelo = tf.keras.models.load_model('Models/Modelo3/Modelo3.h5')
plus=25

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        height,width,c = frame.shape
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        i=0
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                i=i+1
                lista =[]
                for index in index_list:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    lista.append([x,y])
                x1,y1= lista[0]
                x2,y2= lista[1]
                x3,y3= lista[2]
                x4,y4= lista[3]
                longitud1 = math.hypot(x2-x1,y2-y1)
                cv2.putText(frame,f"Persona{i}",(x3-plus,y1-plus-5),1,1.5,colors[i],2)
                cv2.rectangle(frame,(x3-plus,y1-plus),(x4+plus,y2+plus),(255,0,0),2)
                if y1-plus < 0 or x3-plus < 15 or y2+plus > height or x4+plus > width : #condicion de que el rosto no esta cortado por el campo de la camara
                    pass#print("No se detecta el rostro")
                else:
                    if longitud1 < 180: # controla la ditancia de la persona y la camara
                        cv2.putText(frame,"Demaciado Lejos",(x3-plus,y1-plus-5),1,1.5,(255,0,0),2)
                        cv2.rectangle(frame,(x3-plus,y1-plus),(x4+plus,y2+plus),(255,0,0),2)
                    elif longitud1 > 380:  # controla la ditancia de la persona y la camara
                        cv2.putText(frame,"Demaciado Cerca",(x3-plus,y1-plus-5),1,1.5,(255,0,0),2)
                        cv2.rectangle(frame,(x3-plus,y1-plus),(x4+plus,y2+plus),(255,0,0),2)
                    else:  # se procesa la imagen en el modelo
                        interes = frame[y1-plus:y2+plus,x3-plus:x4+plus]
                        interes = cv2.cvtColor(interes, cv2.COLOR_BGR2GRAY)
                        interes = cv2.resize(interes, (48, 48), interpolation=cv2.INTER_AREA)
                        interes = interes.reshape((1,48,48))
                        resultado = modelo.predict(interes)
                        resultado =np.round(resultado[0])
                        t = np.argmax(resultado)
                        #print(t)
                        seleccionar(frame,t,y1,y2,x3,x4,plus)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()