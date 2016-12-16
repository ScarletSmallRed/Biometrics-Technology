1)
from PIL import Image
img = Image.open('lena.png')

cut = (100, 100, 400, 400)
region = img.crop(cut)
img_rotate45 = region.rotate(45)
img_rotate45.show()


2)
import matplotlib.pyplot as plt
import numpy as np

lut = []
lut = img.histogram()
lut
len(lut)
left = []
left = [i for i in range(0, len(lut))]
left
plt.bar(left=left, height=lut, width=0.96)
plt.show()

from PIL import Image
from PIL import ImageFilter


#Perform Max Filtering, Min Filtering, and Median Filter on lena.png.
img = Image.open('lena.png')  # 打开图片

img1 = img.filter(ImageFilter.MinFilter)
img1.show()

img2 = img.filter(ImageFilter.MaxFilter)
img2.show()

img3 = img.filter(ImageFilter.MedianFilter)
img3.show()


#Perform Gaussian Blur with sigma equal to 3 and 5.
img4 = img.filter(ImageFilter.GaussianBlur(radius= 3))
img4.show()

img5 = img.filter(ImageFilter.GaussianBlur(radius= 5))
img5.show()











from PIL import ImageFilter
import math
import numpy as np
from PIL import Image


class MyGaussianBlur():
    # initialize
    def __init__(self, radius=1, sigema=1.5):
        self.radius = radius
        self.sigema = sigema

    def calc(self, x, y):
        res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
        res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
        return res1 * res2
        # get the filter template

    def template(self):
        sideLength = self.radius * 2 + 1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i, j] = self.calc(i - self.radius, j - self.radius)
        all = result.sum()
        return result / all
        # filter function

    def filter(self, image, template):
        arr = np.array(image)
        height = arr.shape[0]
        width = arr.shape[1]
        newData = np.zeros((height, width))
        for i in range(self.radius, height - self.radius):
            for j in range(self.radius, width - self.radius):
                t = arr[i - self.radius:i + self.radius + 1, j - self.radius:j + self.radius + 1]
                a = np.multiply(t, template)
                newData[i, j] = a.sum()
        newImage = Image.fromarray(newData)
        return newImage


r = 2  # radius
s = 3  # sigema(3 or 5, up to the programmer)
GBlur = MyGaussianBlur(radius=r, sigema=s)
temp = GBlur.template()  # get the filter template
img = Image.open('lena.png')  # open the image
image = GBlur.filter(img, temp)
image.show()


