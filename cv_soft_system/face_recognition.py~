import numpy as np
import cv2
import sys

database_path = 'database/' + str(sys.argv[1]) + '/'
print (database_path)
img_list = []
label_list = []

DEF_SIZEX = 200
DEF_SIZEY = 200

##CREATE INPUT FOR RECOGNIZER                                   
for i in range(0, 9):
    img_list.append(cv2.imread(database_path + str(i) + '.pgm'))
    h,w,c = img_list[i].shape
    img_list[i] = cv2.resize(img_list[i], None, fx=DEF_SIZEX/w, fy=DEF_SIZEY/h, interpolation = cv2.INTER_CUBIC)
    label_list.append(int(2))

np_label_list = np.asarray(label_list, dtype=int)

cv2.imshow('teste', img_list[2])
print(str(img_list[2].shape))
cv2.waitKey()

recognizer = cv2.face.createEigenFaceRecognizer()
recognizer.train(img_list, np_label_list)

face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

i = 0;

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.equalizeHist(imgGray)
    faces = face_cascade.detectMultiScale(imgGray, 1.3, 5)

    for (x,y,w,h) in faces:
        face_img = imgGray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, None, fx=DEF_SIZEX/w, fy=DEF_SIZEY/h, interpolation = cv2.INTER_CUBIC)
        founded_person = recognizer.predict(face_img)
        print("FOUND: " + str(founded_person))
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
