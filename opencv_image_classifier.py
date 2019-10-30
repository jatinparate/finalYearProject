import os
import cv2
import numpy as np

faces = []
labels = []
class_names = []

for item in os.listdir('images'):
    class_names.append(item)

for user in os.listdir('images'):
    for item in os.listdir('images/' + user):
        faces.append(cv2.imread('images/' + user + '/' + item, 0))
        labels.append(class_names.index(user))

# create user face classifier
face_classifier = cv2.face.LBPHFaceRecognizer_create()
face_classifier.train(faces, np.array(labels))

# create face recognizer
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

# get the test image
test_image = cv2.imread('test.jpg', 0)

# detect faces
faces = face_cascade.detectMultiScale(test_image, 1.3, 5)

# predict each face
for (x,y,w,h) in faces:
    face = test_image[y:y+h, x:x+w]
    label = face_classifier.predict(face)
    cv2.rectangle(test_image, (x,y), (x+w, y+h), (255, 0, 0), 3)
    cv2.putText(test_image, class_names[label[0]], (x,y), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 5)

# cv2.imshow('output', test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('./out.jpg', test_image)