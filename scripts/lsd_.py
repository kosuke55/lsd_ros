import cv2
import numpy as np

# from pylsd.lsd import lsd


lsd = cv2.createLineSegmentDetector()




img = cv2.imread('box_zoom.png')
img2 = img.copy()
img3 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 5)
lines = lsd.detect(gray)[0]
lsd.drawSegments(img, lines)

# print(lines)
# print(lines.shape)
for line in lines:
    print(line[:4])
    print(line.shape)
    x1, y1, x2, y2 = map(int, line[0])
    img2 = cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 3)
    if (x2-x1)**2 + (y2-y1)**2 > 1000:
        img3 = cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 3)

print(img.shape)
cv2.imwrite('result.jpg', img)
cv2.imwrite('samp_pylsd.jpg', img2)
cv2.imwrite('samp_pylsd2.jpg', img3)
