import cv2
import numpy as np

b1 = []
for i in range(0, 30):
    b1.append(0)

b2 = []
for i in range(0, 640):
    b2.append(b1)

b3 = []
for k in range(0, 480):
    b3.append(b2)
#################################
g1 = []
for i in range(0, 30):
    g1.append(0)

g2 = []
for i in range(0, 640):
    g2.append(b1)

g3 = []
for k in range(0, 480):
    g3.append(b2)
################################
r1 = []
for i in range(0, 30):
    r1.append(0)

r2 = []
for i in range(0, 640):
    r2.append(r1)

r3 = []
for k in range(0, 480):
    r3.append(r2)
################################
b3 = np.asarray(b3)
g3 = np.asarray(g3)
r3 = np.asarray(r3)

cap=cv2.VideoCapture('traffic.mp4')
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for fr in range(0, 30):
    success,img=cap.read()
    print(success)
    print(b3[0][0])
    print(img[0][0][0])
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            b3[i][j][fr] = img[i][j][0]
            g3[i][j][fr] = img[i][j][1]
            r3[i][j][fr] = img[i][j][2]

sImg0 = []
for i in range(0, 640):
    sImg0.append([0, 0, 0])

sImg = []
for j in range(0, 480):
    sImg.append(sImg0)
sImg = np.asarray(sImg)

b3 = np.float32(b3)
g3 = np.float32(g3)
r3 = np.float32(r3)
sImg = np.float32(sImg)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2

for i in range(0, height):
    for j in range(0, width):
        ret, label, centerB = cv2.kmeans(b3[i][j], K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        ret, label, centerG = cv2.kmeans(g3[i][j], K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        ret, label, centerR = cv2.kmeans(r3[i][j], K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        b = max(centerB)
        g = max(centerG)
        r = max(centerR)
        sImg[i][j][0] = b
        sImg[i][j][1] = g
        sImg[i][j][2] = r
print("**********************************")

cv2.imshow('img_final', sImg)
cv2.waitKey(0)
