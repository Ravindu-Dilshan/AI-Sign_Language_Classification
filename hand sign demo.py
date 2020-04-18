# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:18:17 2020

@author: user
"""
import cv2
import numpy

from keras.models import load_model

model = load_model('hand_sign.h5')
CLASS_MAP = {"A": 0,"B": 1,"C": 2,"D": 3,"E": 4,"F": 5,"G": 6,"H": 7,"I": 8,"K": 9,
             "L": 10,"M": 11,"N": 12,"O": 13,"P": 14,"Q": 15,"R": 16,"S": 17,"T": 18,"U": 19,
             "V": 20,"W": 21,"X": 22,"Y": 23,"nothing": 24}


CLASS_MAP_REV = dict(map(reversed, CLASS_MAP.items()))
def mapper(val):
    return CLASS_MAP[val]

def mapper_rev(val):
    return CLASS_MAP_REV[val]

def identify(image):  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict([image])[0]
    key = mapper_rev(numpy.argmax(pred))
    acc = str(int(max(pred)*100))+"%"
    return key,acc

detect = False
letterDetected = "NONE"

# Create a window to display the camera feed
cv2.namedWindow('Camera Output')
cv2.namedWindow('Hand')
# Get pointer to video frames from primary device

WIDTH, HEIGHT = (1000,1000)
FPS = 30
videoFrame = cv2.VideoCapture(0)
videoFrame.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH);
videoFrame.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT);
videoFrame.set(cv2.CAP_PROP_FPS,FPS);

while True:

    readSucsess, frame = videoFrame.read()
    cv2.rectangle(frame, (150, 50), (400, 300), (255, 255, 255), 2)
    cv2.rectangle(frame, (550, 50), (800, 300), (255, 255, 255), 2)
    sourceImage = frame[50:300, 150:400]
    #handImage = sourceImage.copy()[10:300, 10:300]
    if(detect == True):
        letterDetected, acc = identify(sourceImage)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Letter: " + letterDetected,
                (150, 350), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Accuracy: " + acc,
                (550, 350), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Place Hand on Left Box",
                (0, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Camera Output', frame)
    cv2.imshow('Hand', sourceImage)

    keyPressed = cv2.waitKey(30)
    if keyPressed == 27:
        break
    if keyPressed == 32:
        if(detect):
            detect = False
        else:
            detect = True

cv2.destroyWindow('Camera Output')
cv2.destroyWindow('Hand')
videoFrame.release()
