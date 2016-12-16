# Assignment 4: Background Recovery

In this task, you will deal with videos with OpenCV. The attached trafic.mp4 is a video taken by stationary cameras. You will be instructed to recover background by two methods.

## Method 1: Averaging

* Read the mp4 video.

```python
cap= cv2.VideoCapture(’ traffic .mp4’)
```

* Print the frame width, frame height, frames per second and frame count of the input video.

```python
cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
cap.get(cv2.CAP_PROP_FRAME_FPS) 
cap.get(cv2.CAP_PROP_FRAME_COUNT)
```

* Convert frame width, frame height, frames per second and frame count into integers using int().
* Get the background object by averaging away the foreground (i.e. moving) objects using the following suggested codes: 

```python
img = cap.get() 
avgImg = np.float32(img) 
for fr in range(1, frameCount): 
	img = cap.read() 
	write your code here 

cap.release () 
```

* Capture the background and save the background image.  

## Method 2: Color distribution

Read Page5-13 in removebg.pdf and implement it. In this method, you will perform clustering for each pixel over all the frames of the video. For more detail, you may follow these two steps for each pixel(i,j) in the image:

1. Collect the BGR values of pixel (i, j) for all frames.2. Use k-means method to perform clustering on these BGR values. Use cv
2. kmeans() with 2 cluster centers. The center of larger cluster is then the desired BGR value for pixel (i,j) of background image.Save the background image.

## Questions

1. Compare the two methods. Which method gives a better result? Why?
2. How would you use SVD to recover the background?