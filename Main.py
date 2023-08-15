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
        #---------------------declaracion de variables-----------------------
        self.viaje=True
        self.ui = UI()
        self.ui.setupUi(self)
        self.cap = cv2.VideoCapture(0)
        width = 950
        height = 650
        self.numpersonas =1
        frame_inity = np.zeros((height,width))
        frame_inity[frame_inity==0]=200
        image_qt = QImage(frame_inity.data, frame_inity.shape[1], frame_inity.shape[0], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image_qt)
        self.ui.label_video.setText("")
        # Configura el tamaño de la captura
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
        self.Cargarmodelo()
        self.ObtenerNumPersonas
        self.thread = Proceso(self.estado,self.cap,self.modelo,self.numpersonas)
        self.thread.data_ready.connect(self.pintar)
        self.thread.start()
        

    def pintar(self,data):
        self.ui.label_video.setPixmap(data)
        self.ui.label_video.update()

    def Detener(self):
        self.estado= False
        self.ui.label_video.clear()
        self.ui.label_video.setText("")
        self.ui.label_video.update()
        self.thread.stop()

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
class Proceso(QThread):
    data_ready = pyqtSignal(object)
    def __init__(self,estado,cap,modelo,num):
        super().__init__()
        self.estado = estado
        self.cap = cap
        self.numP = num
        self.colors=[
            (255, 0, 0), # Rojo (Red)
            (128, 0, 128), # Púrpura (Purple)
            (0, 255, 255), # Cian (Cyan)
            (0, 255, 0), # Verde (Green)
            (0, 0, 0), # negro (Black)
            (0, 0, 255), # azul (Blue)
            (255, 255, 0), # Amarillo (Yellow)
            ]
        #Cargamos el modelo
        self.modelo = modelo
        self.mpDibujo = mp.solutions.drawing_utils
        self.configDibujo = self.mpDibujo.DrawingSpec(thickness=1,circle_radius=1)
        self.configDibujo2 =self. mpDibujo.DrawingSpec(thickness=1)
        self.mpMallafacila=mp.solutions.face_mesh
        self.mallaFacila = self.mpMallafacila.FaceMesh(max_num_faces=self.numP)


    def run(self):
        while self.estado:
                rst,frame = self.cap.read()
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                resultado = self.mallaFacila.process(frame)
                px=[]
                py=[]
                lista=[]
                r=5
                t=3
                if resultado.multi_face_landmarks:
                    print(len(resultado.multi_face_landmarks))
                    for rostro in  resultado.multi_face_landmarks:
                        #self.mpDibujo.draw_landmarks(frame,rostro,self.mpMallafacila.FACEMESH_CONTOURS,self.mpDibujo.DrawingSpec(thickness=1,circle_radius=1),self.mpDibujo.DrawingSpec(thickness=1))
                        for id,punto in enumerate(rostro.landmark):
                            al,an,c = frame.shape
                            x,y = int(punto.x*an),int(punto.y*al)
                            px.append(x)
                            py.append(y)
                            lista.append([id,x,y])
                            if len(lista) == 468:
                                plus =25
                                x1,y1 = lista[10][1:]
                                x2,y2 = lista[152][1:]
                                x3,y3 = lista[234][1:]
                                x4,y4 = lista[454][1:]
                                x5,y5 = lista[1][1:]
                                cx,cy = (x1+x2)//2,(y1+y2)//2
                                longitud1 = math.hypot(x2-x1,y2-y1)
                                #cv2.circle(frame,lista[10][1:],5,(255,0,0),3)#arriba
                                #cv2.circle(frame,lista[234][1:],5,(255,0,0),3)#derecha
                                #cv2.circle(frame,lista[152][1:],5,(255,0,0),3)#abajo
                                #cv2.circle(frame,lista[454][1:],5,(255,0,0),3)#izquierda
                                #cv2.circle(frame,lista[1][1:],5,(255,0,0),3)
                                if y1-plus < 0 or x3-plus < 15 or y2+plus > al or x4+plus > an : #condicion de que el rosto no esta cortado por el campo de la camara
                                    pass#print("No se detecta el rostro")
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
                                        #print(t)
                                        self.seleccionar(frame,t,y1,y2,x3,x4,plus)
                else:
                    pass#print("no hay cara")
                image_qt = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image_qt)
                self.data_ready.emit((pixmap))
                t =cv2.waitKey(1)
                if t == 32 or self.estado==False:
                    break
        self.cap.release()

    def stop(self):
        self.quit()
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
            cv2.rectangle(imagen,(x3-plus,y1-plus),(x4+plus,y2+plus),(117,29,83),2)
            cv2.putText(imagen,"Persona detectada",(x3-plus,y1-plus-5),1,1.5,(117,29,83),2)



def main():
    windows = QApplication(sys.argv)
    ventana= Aplicacion()
    ventana.show()

    sys.exit(windows.exec_())

if __name__ == "__main__":
    main()