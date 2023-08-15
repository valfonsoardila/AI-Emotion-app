import os
import cv2
import numpy as np
import random as rn
from tqdm import tqdm

imag = cv2.imread('Application/logo2.jpeg')
img = cv2.resize(imag,(150,150),interpolation=cv2.INTER_AREA)

cv2.imshow("",img)
cv2.imwrite("Application/logo2.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()