import cv2
import math
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf


class mallaFacial():
    def __init__(self):
        super().__init__()
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    t=0
    colors=[
   (0, 0, 255), # Azul (Blue)
    (128, 0, 128), # Púrpura (Purple)
    (0, 255, 255), # Cian (Cyan)
    (0, 255, 0), # Verde (Green)
   (128, 128, 128), # Gris (Gray)
    (255, 0, 0), # Rojo (Red)
    (255, 255, 0), # Amarillo (Yellow)
]
    #Cargamos el modelo
    modelo = tf.keras.models.load_model("Models/Modelo3/Modelo3.h5")
    mpDibujo = mp.solutions.drawing_utils
    configDibujo = mpDibujo.DrawingSpec(thickness=1,circle_radius=1)
    configDibujo2 = mpDibujo.DrawingSpec(thickness=1)
    mpMallafacila=mp.solutions.face_mesh
    mallaFacila = mpMallafacila.FaceMesh(max_num_faces=1)

    def inicar(self):
        while True:
            rst,frame = self.cap.read()
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            resultado = self.mallaFacila.process(frameRGB)
            px=[]
            py=[]
            lista=[]
            r=5
            t=3
            if resultado.multi_face_landmarks:
                for rostro in  resultado.multi_face_landmarks:
                    ####mpDibujo.draw_landmarks(frame,rostro,mpMallafacila.FACEMESH_CONTOURS,mpDibujo.DrawingSpec(thickness=1,circle_radius=1),mpDibujo.DrawingSpec(thickness=1))
                    pass
                for id,punto in enumerate(rostro.landmark):
                    al,an,c = frame.shape
                    x,y = int(punto.x*an),int(punto.y*al)
                    px.append(x)
                    py.append(y)
                    lista.append([id,x,y])
                    #cv2.putText(frame,f"{id}",(x,y),1,1,(255,0,0),1)
                    if len(lista) == 468:
                        #cv2.circle(frame,lista[10][1:],5,(255,0,0),3)#arriba
                        #cv2.circle(frame,lista[234][1:],5,(255,0,0),3)#derecha
                        #cv2.circle(frame,lista[152][1:],5,(255,0,0),3)#abajo
                        #cv2.circle(frame,lista[454][1:],5,(255,0,0),3)#izquierda
                        #cv2.circle(frame,lista[1][1:],5,(255,0,0),3)
                        plus =25
                        x1,y1 = lista[10][1:]
                        x2,y2 = lista[152][1:]
                        x3,y3 = lista[234][1:]
                        x4,y4 = lista[454][1:]
                        x5,y5 = lista[1][1:]
                        cx,cy = (x1+x2)//2,(y1+y2)//2
                        longitud1 = math.hypot(x2-x1,y2-y1)
                        if y1-plus < 0 or x3-plus < 15 or y2+plus > al or x4+plus > an : #condicion de que el rosto no esta cortado por el campo de la camara
                                print("No se detecta el rostro")
                            #     #time.sleep(3)
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
                                resultado = self.modelo.predict(interes)
                                resultado =np.round(resultado[0])
                                t = np.argmax(resultado)
                                print(t)
                                self.seleccionar(frame,t,y1,y2,x3,x4,plus)

                        t= cv2.waitKey(1)
                        # print(f"longitud: {longitud1}")
                        # interes = frame[y1-plus:y2+plus,x3-plus:x4+plus]
                        # print(f"top:{x1,y1}")
                        # print(f"botton:{x2,y2}")
                        # print(f"rigth:{x3,y3}")
                        # print(f"left:{x4,y4}")

                        if t == 32:
                            interes = frame[y1-plus:y2+plus,x3-plus:x4+plus]
                            interes = cv2.cvtColor(interes, cv2.COLOR_BGR2GRAY)
                            interes = cv2.resize(interes, (48, 48), interpolation=cv2.INTER_AREA)
                            interes = interes.reshape((1,48,48))
                            resultado = self.modelo.predict(interes)
                            resultado =np.round(resultado[0])
                            t = np.argmax(resultado)
                            #print(t)
                            self.seleccionar(frame,t,y1,y2,x3,x4,plus)
                            #cv2.imwrite("captura.png",interes)

                        # #ceja derecha
                        # x1,y1 = lista[65][1:]
                        # x2,y2 = lista[158][1:]
                        # cx,cy = (x1+x2)//2,(y1+y2)//2
                        # longitud1 = math.hypot(x2-x1,y2-y1)

                        # #ceja izquierda
                        # x3,y3 = lista[295][1:]
                        # x4,y4 = lista[385][1:]
                        # cx2,c2 = (x3+x4)//2,(y3+y4)//2
                        # longitud2 = math.hypot(x4-x3,y4-y3)

                        # #boca extremos
                        # x5,y5 = lista[78][1:]
                        # x6,y6 = lista[388][1:]
                        # cx3,c3 = (x5+x6)//2,(y5+y6)//2
                        # longitud3 = math.hypot(x6-x5,y6-y5)

                        # #boca apertura
                        # x7,y7 = lista[13][1:]
                        # x8,y8 = lista[14][1:]
                        # cx4,cy4 = (x7+x8)//2,(y7+y8)//2
                        # longitud4 = math.hypot(x8-x7,y8-y7)
                        # #print(longitud1,',',longitud2,',',longitud3,',',longitud4,)
                        # if longitud1<19 and longitud2<19 and longitud3>80 and longitud3<95 and longitud4<5:
                        #     cv2.putText(frame,"Persona Enojada",(280,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                        # elif longitud1>20 and longitud1<30 and longitud2>20 and longitud2<30 and longitud3>109 and longitud4>10 and longitud4<20: 
                        #     cv2.putText(frame,"Persona Feliz",(280,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
                        # elif longitud1>35 and longitud2>35 and longitud3>85 and longitud3<90 and longitud4>20:
                        #     cv2.putText(frame,"Persona asombrad",(280,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                        # elif longitud1>25 and longitud1<35 and longitud2>25 and longitud2<35 and longitud3>90 and longitud3<95 and longitud4<5: 
                        #     cv2.putText(frame,"Persona Triste",(280,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            else:
                print("no hay cara")

            cv2.imshow("Reconocimiento de emociones", frame)
            t= cv2.waitKey(1)
            if t==27:
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def seleccionar(self,imagen,clase,y1,y2,x3,x4,plus):
        if clase == 0:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,"Persona Enojada",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        elif clase == 1:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,"Persona Asqueado",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        elif clase == 2:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,"Persona Asustada",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        elif clase == 3:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,"Persona Feliz",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        elif clase == 4:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,"Persona Neutral",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        elif clase == 5:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,"Persona Triste",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        elif clase == 6:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,"Persona Sorprendida",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        else:
            pass

m=mallaFacial()
m.inicar()
