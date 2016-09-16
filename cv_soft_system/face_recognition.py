import numpy as np
import cv2
import sys

database_path = str(sys.argv[1])
print (database_path)

DEF_SIZEX = 200
DEF_SIZEY = 200

def createInputsForRecognizer(database_root_path):
    database_folders = os.listdir(database_root_path)
    list_file = []
    list_id = []
    if len(database_folders) == 0:
        print ("ERROR: Invalid Folder. Make sure that folder is correct and all files in tree are pictures!")
        sys.exit()
    
    for i in range(0, len(database_folders)):
    # Get pictures inside person folder
        files = os.listdir(database_path + database_folders[i] + "/") 
        list_file.extend(files)
        for j in range(0, len(files)):
            list_id.append(i)

    return list_file, list_id


recognizer = cv2.face.createEigenFaceRecognizer()
recognizer.train(createInputsForRecognizer(database_path))
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
