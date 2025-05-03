import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

# Initialize paths and data
path = 'ImagesAtt'
images = []
classNames = []

# Load images and names
if not os.path.exists(path):
    raise Exception(f"Image folder '{path}' not found!")

myList = os.listdir(path)
for cl in myList:
    img_path = os.path.join(path, cl)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Failed to load image {cl}")
        continue
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])

# Encode known images
def find_encodings(images):
    encodeList = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

lastMarkedTime = {}

def markAttendance(name):
    currentTime = datetime.now()
    if name in lastMarkedTime:
        if currentTime - lastMarkedTime[name] < timedelta(minutes=1):
            return
    lastMarkedTime[name] = currentTime
    with open('Att.csv', 'a+') as f:
        f.write(f'\n{name},{currentTime.strftime("%H:%M:%S")}')

encodeListKnown = find_encodings(images)
print(f"[INFO] Encoding completed for {len(encodeListKnown)} faces.")

# Start webcam
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    raise Exception("Webcam not accessible. Check if it's connected or available.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(imgS)
    face_encodings = face_recognition.face_encodings(imgS, face_locations)

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        name = 'Unknown'

        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()
            markAttendance(name)

        y1, x2, y2, x1 = [val * 4 for val in faceLoc]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam - Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
