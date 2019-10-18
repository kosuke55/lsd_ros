import cv2
import time
# from pylsd.lsd import lsd


lsd = cv2.createLineSegmentDetector()


import numpy as np

img = cv2.imread('box_zoom.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),5)

t1 = time.time()
t2 = time.time()

linesL = lsd(gray)
t3 = time.time()

img2 = img.copy()
cv2.imwrite('samp_hagh.jpg',img2)
img3 = img.copy()
img4 = img.copy()
for line in linesL:
    x1, y1, x2, y2 = map(int,line[:4])
    img3 = cv2.line(img3, (x1,y1), (x2,y2), (0,0,255), 3)
    if (x2-x1)**2 + (y2-y1)**2 > 1000:
       img4 = cv2.line(img4, (x1,y1), (x2,y2), (0,0,255), 3)

cv2.imwrite('samp_pylsd.jpg',img3)
cv2.imwrite('samp_pylsd2.jpg',img4)
