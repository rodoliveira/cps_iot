import numpy as np
import cv2
import sys
import os

database_path = str(sys.argv[1])
print (database_path)

DEF_SIZEX = 200
DEF_SIZEY = 200

def preProcessPicture(f_img):
    fh, fw = f_img.shape
    proc_img = cv2.equalizeHist(f_img)
    proc_img = cv2.resize(f_img, None, fx=DEF_SIZEX/fw, fy=DEF_SIZEY/fh, interpolation = cv2.INTER_CUBIC)
    return proc_img
    
def createInputsForRecognizer(database_root_path):
    database_folders = os.listdir(database_root_path)
    list_img = []
    list_id = []
    list_idLabel = []
    
    if len(database_folders) == 0:
        print ("ERROR: Invalid Folder. Make sure that folder is correct and all files in tree are pictures!")
        sys.exit()
    
    for i in range(0, len(database_folders)):
    # Get pictures inside person folder
        files = os.listdir(database_path + database_folders[i] + "/")
        list_idLabel.append(database_folders[i])
        for j in range(0, len(files)):
            f_img = cv2.imread(database_path + database_folders[i] + "/" + files[j], 0)
            list_img.append(preProcessPicture(f_img))
            list_id.append(i)

        np_list = np.asarray(list_id, dtype=int)

    return list_img, np_list, list_idLabel

img_list, label_list, person_list = createInputsForRecognizer(database_path)
recognizer = cv2.face.createEigenFaceRecognizer()
recognizer.train(img_list, label_list)
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
        founded_person_id = recognizer.predict(face_img)
        print("FOUND: " + person_list[founded_person_id])
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
