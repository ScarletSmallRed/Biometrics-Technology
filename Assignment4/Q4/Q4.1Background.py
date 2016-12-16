import cv2
import numpy as np
from PIL import Image

cap=cv2.VideoCapture('traffic.mp4')
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sImg0=[]
for i in range(0,640):
    sImg0.append([0,0,0])
    
sImg=[]
for j in range(0,480):
    sImg.append(sImg0)
sImg = np.asarray(sImg)


for fr in range(0, 300):
    success,img=cap.read()
    print(success)
    print(sImg[0][0][0])
    print(img[0][0][0])
    for i in range(0, len(sImg)):
        for j in range(0, len(sImg[0])):
             for k in range(0, len(sImg[0][0])):
                img_num = img[i][j][k]
                sImg[i][j][k] += img_num
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

for i in range(0, len(sImg)):
    for j in range(0, len(sImg[0])):
        for k in range(0, len(sImg[0][0])):
            sImg[i][j][k] /= 300

print("********************************")

cv2.imshow('img_final', sImg)
cv2.imwrite('img_final.png', sImg)
cv2.waitKey(0)
# img = Image.open('img_final.png')
