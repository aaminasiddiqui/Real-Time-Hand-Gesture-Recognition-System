import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
from cvzone.ClassificationModule import Classifier
import tensorflow


labels = ['A', 'B', 'C']
offset = 20
imgSize = 300
counter = 0


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/Keras_model.h5","Model/labels.txt")
while True:

    success, img = cap.read()
    imgCopy = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        #cropping
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        # imgCropShape = imgCrop.shape

        aspectratio = h / w
        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize  #overlaying
            prediction,index=classifier.getPrediction(imgWhite ,draw=False)
            print (prediction,index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize  #overlaying
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        cv2.rectangle(imgCopy,
                      (x-offset,y-offset-50),
                      (x-offset+90,y-offset-50+50),
                      (0,255,0),cv2.FILLED)
        cv2.putText(imgCopy,labels[index],(x,y-20),
                    cv2.FONT_HERSHEY_COMPLEX,1.7,(0,0,0),2)
        #box on hand
        cv2.rectangle(imgCopy, (x-offset,y-offset), (x+w-offset,y+h-offset), (0,255,0),4)

        cv2.imshow("ImgCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)
    cv2.imshow("Image", imgCopy)
    key = cv2.waitKey(1)

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
