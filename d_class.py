import cv2
from mask import *
#import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')

class Detector():
    def __init__(self):
        pass
    def detect_face(img, gray):
        faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=15,
        minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
       
    def detect_eyes(img, gray):
        eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(30, 30)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
    def detect_mouth(img, gray):
        mouth = mouth_cascade.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=20,
        minSize=(40, 40)
        )
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(img, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
    def mask(img, gray):
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=15,minSize=(30, 30))
        for (x, y, w, h) in faces:
            overlay_sunglasses(img,x,y,w,h)
