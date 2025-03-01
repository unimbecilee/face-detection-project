#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 08:51:37 2025

@author: unimbecileee
"""

import cv2 as cv

# Charger la cascade de Haar pour la détection de visages
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Charger les images
img = cv.imread('/Volumes/Landry/Projet-ai/face-detect/obama.jpg')

# Vérifier si l'image a été chargée correctement
if img is None:
    print("Erreur : Impossible de charger l'image. Veuillez vérifier le chemin et le nom de fichier.")
else:
    # Convertir l'image en niveaux de gris
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Exécution de la détection de visage
    faces = face_cascade.detectMultiScale(gray, 1.1, 8)

    # Afficher les visages
    i = 0
    for face in faces:
        x, y, w, h = face
        
        # dessiner le rectangle sur l'image principale
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Etraire les visages de l'image principale
        # OpenCV et Numpy: y <-> ligne et x <-> colone
        face = img[y:y+h, x:x+w]
        
        # Afficher face0, face1, face2, etc...
        cv.imshow('face{}'.format(i), face)
        i += 1

# Afficher l'image avec les visages détectés
cv.imshow('image pincipale', img)
cv.waitKey(0)
cv.destroyAllWindows()
