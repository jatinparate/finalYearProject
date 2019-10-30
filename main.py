import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('./test.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.5, 5)

for (x, y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 5)


cv2.imwrite('./out.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()