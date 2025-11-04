import cv2
import numpy as np
from fonctions_fusion import *



cap = cv2.VideoCapture(0)
frame_count = 0
sizes = []
grilles = Grid()

while True:
    ret, frame = cap.read()
    if not ret:
        print("La vidéo est terminée ou vide.")
        break
    
    

    cropped_frame, vals = grilles.detect_black_rectangles(frame)
    print(vals)
    print(" ")
    
    cv2.imshow("frame", cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

