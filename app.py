# -*- coding: utf-8 -*-


import cv2 as cv
import sys

# Charger le classificateur en cascade pré-entrainé
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Charger les images
img = cv.imread('/Volumes/Landry/Projet-ai/face-detect/obama.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Exécution de la détection de visage
faces = face_cascade.detectMultiScale(img_gray, 1.1, 8)

# Verifier le nopmbre de visages 
if len(faces) != 2:
    sys.exit('La photo doit avoir exactement 2 visages, réessayez svp...')
    
# Recuperation des dimensions de chaque visage
x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

# Extraction des 2 visages de l'image
face1 = img[y1:y1+h1, x1:x1+w1]
face2 = img[y2:y2+h2, x2:x2+w2]

# Redimensionner face2 aux dimensions de face1 et vice versa
face2 = cv.resize(face2, (w1,h1))
face1 = cv.resize(face1, (w2,h2))


# Remplacer face2 par face1
img[y2:y2+h2, x2:x2+w2] = face1

# Remplacer face1 par face2
img[y1:y1+h1, x1:x1+w1] = face2

# Affichage des visages
cv.imshow('echange', img)
cv.waitKey(0)
cv.destroyAllWindows()