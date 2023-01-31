import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


## for Videos
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),2)
    cv2.imshow("MV Face Detector",frame)
    key = cv2.waitKey(1)
    # Q for Quit
    if key==81 or key==113:
        break

webcam.release()







### for Images
"""

img = cv2.imread("demo.jpg")

grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),2)


cv2.imshow("MV Face Detector",img)
cv2.waitKey()
"""
print("Complete")