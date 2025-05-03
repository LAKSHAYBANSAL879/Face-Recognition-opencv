import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import mss

# Initialize paths and data
path = 'ImagesAtt'
images = []
classNames = []
myList = os.listdir(path)

# Load and encode images for known faces
for cl in myList:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Dictionary to track the last attendance time for each person
lastMarkedTime = {}

def markAttendance(name):
    currentTime = datetime.now()
    # Only mark attendance if the last mark was more than a minute ago
    if name in lastMarkedTime:
        lastTime = lastMarkedTime[name]
        if currentTime - lastTime < timedelta(minutes=1):
            return  # Skip marking within 1 minute
    
    # Update the last marked time and log to file
    lastMarkedTime[name] = currentTime
    with open(r'D:\DSML\Opencvv\Att.csv', 'a+') as f:
        f.writelines(f'\n{name},{currentTime.strftime("%H:%M:%S")}')

# Encode known faces
encodeListKnown = find_encodings(images)
print(f"Encoding completed for {len(encodeListKnown)} faces.")

# Set up screen capture with mss
with mss.mss() as sct:
    monitor = sct.monitors[1]

    # Main loop for face recognition and frame display
    while True:
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodedFace, FaceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodedFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodedFace)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] < 0.50:
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                name = 'Unknown'

            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Display the single updated frame in a single window
        cv2.imshow('Screen Capture', img)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()