import cv2
import numpy as np
from PIL import Image

img = cv2.imread('lena.png', 0)

Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

from scipy import signal
Mx = signal.convolve2d(img, Gx, mode="same")
My = signal.convolve2d(img, Gy, mode="same")

M = np.sqrt(Mx * Mx + My * My)
print(M)

M = 255 * M / np.max(M)
M = np.array(M, dtype = 'uint8')
img1 = Image.fromarray(M)
img1.show()
