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

v0 = np.array([0, 1.0])

for line in lines:
    # print(line[0])
    x1, y1, x2, y2 = map(int, line[0])
    v = np.array([x2-x1, y2 - y1])
    ve = v / np.linalg.norm(v)
    dot = np.dot(v0, ve)
    print(np.abs(dot))
    img2 = cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 3)
    if ((x2-x1)**2 + (y2-y1)**2 > 2000 and np.abs(dot) > 0.9):
        img3 = cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 3)

cv2.imwrite('result.jpg', img)
cv2.imwrite('result2.jpg', img2)
cv2.imwrite('resulr3.jpg', img3)
