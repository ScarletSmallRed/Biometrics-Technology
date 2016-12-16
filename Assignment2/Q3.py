import cv2


#read the image
img = cv2.imread('bee.png')


#Convert the image to HSV color space.
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


#Perform histogram equalization on V channel by cv2.equalizeHist()
H, S, V = cv2.split(img_hsv)
equ_V = cv2.equalizeHist(V)

img_merge = cv2.merge([H, S, equ_V])
# cv2.imshow('img_merge', img_merge)
# cv2.imwrite('img_merge.png', img_merge)
# cv2.waitKey(0)

import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('img_merge.png')
lut = []
lut = img.histogram()
lut
len(lut)
left = []
left = [i for i in range(0, len(lut))]
left
plt.bar(left = left, height = lut, width = 0.96)
plt.show()


#Convert the result image to BGR color space.
img_merge_bgr = cv2.cvtColor(img_merge, cv2.COLOR_HSV2BGR)


# Show the image by cv2.imshow() and save the image.
cv2.imshow('img_merge_bgr', img_merge_bgr)
cv2.imwrite('img_merge_brg.png', img_merge_bgr)
cv2.waitKey(0)
