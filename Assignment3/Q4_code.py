from PIL import Image
from PIL import ImageOps
import numpy
from numpy import linalg
import matplotlib
from matplotlib import pyplot
flower=Image.open("C:/Users/wang/Desktop/hw3/flower.jpg")
flower.show()
flower=ImageOps.grayscale(flower)
flower.show()
aflower=numpy.asarray(flower)
aflower=numpy.float32(aflower)
U,S,Vt=linalg.svd(aflower)
pyplot.plot(S,'b.')
pyplot.show()
K=20
Sk=numpy.diag(S[:K])
Uk=U[:,:K]
Vtk=Vt[:K,:]
aImk=numpy.dot(Uk,numpy.dot(Sk,Vtk))
Imk=Image.fromarray(aImk)
Imk.show()
