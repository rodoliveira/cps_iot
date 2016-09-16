import numpy as np
import cv2
import sys

database_path = '../face_databases/' + str(sys.argv[1]) + '/'
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
count_faces = 0
cap = cv2.VideoCapture(0)

max_faces = 10

scale_fact = 1.0
while(count_faces < max_faces):
    key = cv2.waitKey(1);
    ret, frame = cap.read()
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.equalizeHist(imgGray)
    faces = face_cascade.detectMultiScale(imgGray, 1.3, 5)

    for (x,y,w,h) in faces:
        if (len(faces) > 0):
          frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          xinit = x-(w*(scale_fact-1))
          yinit = y-(h*(scale_fact-1))
          crop_img = imgGray[yinit:yinit+(h*scale_fact),xinit:xinit+(w*scale_fact)]
          cv2.imshow('teste', crop_img)
          if key & 0xFF == ord('w'):
              cv2.imwrite(database_path + str(count_faces)+'.pgm', crop_img)
              print("IMAGE SAVE: " + str(count_faces))
              count_faces = count_faces + 1;

    cv2.imshow('img',frame)
    
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
