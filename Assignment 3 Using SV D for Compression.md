# Assignment 3: Using SV D for Compression

Stepping gingerly from the spaceship onto Martian soil, you heave a sigh of relief. All these years of training at the School of Cosmology (SOC) has finally paid o↵: you are on your first mission to Mars. As you survey the barren and rocky landscape, your eyes spot something unusual in the distance. You move closer and discoverwhat looks like a Martian flower. Very excitedly, you whip out your high resolution digital camera and take a few shots of the exotic plant.

Back at the spaceship, you wonder how to transmit the images back to Earth. Each image is large, and because of bandwidth limitations, you can send only a small amount of data at a time. How best to do it? Fortunately, you remember the SV D technique. You decide to use it to transmit an image of the flower progressively: a coarse approximation at first, finer details later.

* In Python, import all the necessary libraries as following: 

```python
import Image

import ImageOpsimport numpy     

from numpy import linalg     

import matplotlib     

from matplotlib import pyplot
```

* Read the image using:

```python
flower = Image.open("flower.jpg")
```

* To see the image, use:

```python
flower.show()
```

* This is an RGB image. Convert it to grayscale:

```python
flower = ImageOps.grayscale(flower)
```

* Convert image type to array using:

```python
aflower = numpy.asarray(flower) # aflower is unit8 
aflower = numpy.float32(aflower)
```

* Compute the SVD:

```python
U,S,Vt = linalg.svd(aflower)
```

* The singular values in S have been sorted in descending order. Plot it with the command:

```python
pyplot.plot(S,’b.’)
pyplot.show()
```

* Print out the plot and submit it. What do you notice?
* Let K = 20. Extract the first K singular values and their corresponding vectors in U and V:

```pyton
K = 20
Sk = numpy.diag(S[:K])
Uk = U[:, :K]
Vtk = Vt[:K, :]
```

* Uk, Vk, Sk contain the compressed version of the image. To see this, form the compressed image using:

```python
aImk = numpy.dot(Uk, numpy.dot( Sk, Vtk))
Imk = Image.fromarray(aImk)
Imk.show()
```

* Print out a copy of this and submit it.
* Repeat for K = 50,100,200. Print and submit the compressed images for the four different values of K. Briefly describe what you notice.
* Thus, instead of transmitting the original image, you can transmit Uk, Vk, Sk, which should be much less data than the original. Is it worth transmitting when K = 200 ?