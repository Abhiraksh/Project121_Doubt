# Doing with the black color

from doctest import OutputChecker
import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputFile = cv2.VideoWriter('Output.avi', fourcc, 20.0, (640,480))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cap.read()
bg = np.flip(bg, axis = 1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break 
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerBl1 = np.array([104,152,70])
    upperBl1 = np.array([30,30,0])
    mask1 = cv2.inRange(hsv, lowerBl1, upperBl1)
    lowerBl2 = np.array([255,255,255])
    upperBl2 = np.array([255,255,255])
    mask2 = cv2.inRange(hsv, lowerBl2, upperBl2)

    mask3 = mask1 + mask2
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    mask4 = cv2.bitwise_not(mask3)

    res1 = cv2.bitwise_and(img, img, mask = mask4)
    res2 = cv2.bitwise_and(bg, bg, mask = mask3)
    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)
    outputFile.write(finalOutput)
    cv2.imshow("No Color Black", finalOutput)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()