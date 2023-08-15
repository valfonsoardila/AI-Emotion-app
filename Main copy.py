import sys
import cv2
import math
import time
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Application.Interfaz import Ui_MainWindow as UI
import tensorflow as tf

class Aplicacion(QMainWindow):
    def __init__(self):
        super().__init__()

        self.InicializarGUI()

    def InicializarGUI(self):
        self.colors=[
            (255, 0, 0), # Rojo (Red)
            (128, 0, 128), # Púrpura (Purple)
            (0, 255, 255), # Cian (Cyan)
            (0, 255, 0), # Verde (Green)
            (0, 0, 0), # negro (Black)
            (0, 0, 255), # azul (Blue)
            (255, 255, 0), # Amarillo (Yellow)
            ]
        #---------------------declaracion de variables-----------------------
        self.viaje=True
        self.ui = UI()
        self.ui.setupUi(self)
        self.cap = cv2.VideoCapture(0)
        self.width = 950
        self.height = 650
        self.numpersonas =1
        frame_inity = np.zeros((self.height,self.width))
        frame_inity[frame_inity==0]=200
        image_qt = QImage(frame_inity.data, frame_inity.shape[1], frame_inity.shape[0], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image_qt)
        self.ui.label_video.setText("")
        # Configura el tamaño de la captura
        
        self.modelo = None
        self.estado = True
        self.ui.btn_close.clicked.connect(self.closeApplication)
        self.ui.btn_minimize.clicked.connect(self.minimizeApplication)
        self.ui.btn_on.clicked.connect(self.Iniciar)
        self.ui.btn_off.clicked.connect(self.Detener)
        self.ui.cbox_Modelo.currentTextChanged.connect(self.Cargarmodelo)
        self.ui.cbox_numPers.currentTextChanged.connect(self.ObtenerNumPersonas)
        self.Cargarmodelo()
        self.showMaximized()

    def Cargarmodelo(self):
        if self.ui.cbox_Modelo.currentText()== 'Modelo 1':
            self.modelo = tf.keras.models.load_model('Models/Modelo1/Modelo1.h5')
            self.Summary("Models/Modelo1/Modelo1.txt")
        elif self.ui.cbox_Modelo.currentText()== 'Modelo 2':
            self.modelo = tf.keras.models.load_model('Models/Modelo2/Modelo2.h5')
            self.Summary("Models/Modelo2/Modelo2.txt")
        elif self.ui.cbox_Modelo.currentText()== 'Modelo 3':
            self.modelo = tf.keras.models.load_model('Models/Modelo3/Modelo3.h5')
            self.Summary("Models/Modelo3/Modelo3.txt")
        else:
            pass
    
    def ObtenerNumPersonas(self):
        self.numpersonas = int(self.ui.cbox_numPers.currentText())

    def Iniciar(self):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        mp_face_mesh = mp.solutions.face_mesh
        index_list =[10,152,234,454]
        plus=25
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            min_detection_confidence=0.5) as face_mesh:

            while True:
                ret, frame = self.cap.read()
                
                if ret == False:
                    break
                height,width,c = frame.shape
                frame = cv2.flip(frame,1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)
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
                        if y1-plus < 0 or x3-plus < 15 or y2+plus > height or x4+plus > width : #condicion de que el rosto no esta cortado por el campo de la camara
                            pass#print("No se detecta el rostro")
                        else:
                            if longitud1 < 140: # controla la ditancia de la persona y la camara
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
                                #print(t)
                                self.seleccionar(frame,t,y1,y2,x3,x4,plus,i)
                image_qt = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image_qt)
                self.ui.label_video.setPixmap(pixmap)
                self.ui.label_video.update()
                t =cv2.waitKey(1)
                if t == 32 or self.estado==False:
                    break

    def Detener(self):
        self.cap.release()
        self.estado= False
        self.ui.label_video.clear()
        self.ui.label_video.setText("")
        self.ui.label_video.update()

    def closeApplication(self):
        QApplication.quit()

    def minimizeApplication(self):
        self.showMinimized()

    def Summary(self, file_path):
        try:
            with open(file_path, "r") as file:
                content = file.read()
                self.ui.pt_Info.setPlainText(content)
                self.ui.pt_Info.viewport().update()
        except FileNotFoundError:
            print(f"El archivo '{file_path}' no existe.")
        except IOError:
            print(f"Error al leer el archivo '{file_path}'.")

    def seleccionar(self,imagen,clase,y1,y2,x3,x4,plus,i):
        if clase == 0:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,f"Persona{i} Enojada",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
            
        elif clase == 1:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,f"Persona{i} Asqueado",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
            
        elif clase == 2:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,f"Persona{i} Asustada",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
        elif clase == 3:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,f"Persona{i} Feliz",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
            
        elif clase == 4:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,f"Persona{i} Neutral",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
            
        elif clase == 5:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,f"Persona{i} Triste",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
            
        elif clase == 6:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),self.colors[clase],2)
            cv2.putText(imagen,f"Persona{i} Sorprendida",(x3-plus,y1-plus-5),1,1.5,self.colors[clase],2)
            
        else:
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),(117,29,83),2)
            cv2.putText(imagen,f"Persona{i} detectada",(x3-plus,y1-plus-5),1,1.5,(117,29,83),2)



def main():
    windows = QApplication(sys.argv)
    ventana= Aplicacion()
    ventana.show()

    sys.exit(windows.exec_())

if __name__ == "__main__":
    main()