import numpy as np
import cv2

flagBW = False
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    key = cv2.waitKey(33)
     
    if (flagBW):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord('b'):
        flagBW = not flagBW

    cv2.imshow('img',frame)
    
cap.release()
cv2.destroyAllWindows()
