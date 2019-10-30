import cv2
import numpy
import os

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

test_image = cv2.imread('test.jpg', 0)

faces = face_cascade.detectMultiScale(test_image, 1.3, 5)

for (x,y,w,h) in faces:
    face = test_image[y:y + h, x:x + w]
    cv2.imshow('image', face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    user = input('who is this? : ')
    if (not os.listdir('images').__contains__(user)):
        os.mkdir('images/' + user)
    cv2.imwrite('images/' + user + '/1.jpg', face)

