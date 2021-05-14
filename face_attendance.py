import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path="students"
images=[]
studentNames=[]
myList = os.listdir(path)

#for taking names from database itself
for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    studentNames.append(os.path.splitext(cl)[0])
print(studentNames)

# for rgb conversion and encoding 
def findEncoding (images):
    encodeList = []
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# for marking attendance into the sheet

def markAttendance(name):
     with open('Attendance.csv','r+') as f: 
         myDataList = f.readlines()
         nameList=[]
         for line in myDataList:
             entry=line.split(',') # for taking only name 
             nameList.append(entry[0])

         if name not in nameList:
            now=datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')




encodeListKnow = findEncoding(images)
print("encoding complete")

#  webcam features are used with the help of cv2 library 

cap=cv2.VideoCapture(0)

while True:
    success, img= cap.read()  # img contains the image captured by camera
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)  # we resize it by factor of 0.25 and call it imgs
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)  # step 1 rgb conversion of imgs

    facesCurFrame = face_recognition.face_locations(imgs) # step 2.1 of finding face in an imgs , 4 para tl tr bl br
    encodesCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)  # step 2.2  imgs encoding 

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace) # step 3.1 face comparison of imgs with DB images
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)  # step 3.2.1 calculating distances of imgs with DB images
        matchIndex = np.argmin(faceDis)  # step 3.2.2 minimum the distance the better match is 

        if matches[matchIndex]:
            name = studentNames[matchIndex].upper()  # Upper case name of matched student
            y1,x2,y2,x1=faceLoc                        
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4          # as facelocations were for imgs so we need to divide it by 0.25 i.e mul with 4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)   # show rect with green color
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name) #mark the attendance

    cv2.imshow('webcam',img)
    cv2.waitKey(1)

