from PIL import Image
from PIL import ImageOps
import numpy as np
from numpy import linalg
import matplotlib
from matplotlib import pyplot

flower = Image.open('flower.jpg')
# flower.show()

flower = ImageOps.grayscale(flower)
# flower.show()

aflower = np.asarray(flower)
aflower = np.float32(aflower)

U, S, Vt = linalg.svd(aflower)

pyplot.plot(S, 'b.')
# pyplot.show()

K = 200
Sk = np.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]

aImk = np.dot(Uk, np.dot( Sk, Vtk))
Imk = Image.fromarray(aImk)
Imk.show()
