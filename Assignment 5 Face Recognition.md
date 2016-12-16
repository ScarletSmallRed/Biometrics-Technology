# Assignment 5: Face Recognition

**School of Computing National University of Singapore**

Biometrics Course July 2016

## Goal

The goal in this exercise is to use Principal Components Analysis (PCA) and Linear Discriminating Analysis (LDA) to recognize face images. Read all instructions first before coding. Some planning is required.

## Data

Download face.zip in your computer and extract the images from face.zip. In this, you will find some face images taken from the CMU PIE database. There are 10 people, with each person captured under 24 different lighting conditions, for a total of 240 images. For convenience, these images have been split into two equal sets in the train and test directories.

The image files have names ppppp xx yy.bmp, where ppppp denotes the identity of the person; xx denotes the head orientation (all are frontal (xx=27) in this exercise); and yy denotes the lighting condition. All images have been cropped and aligned, and their height, width are 160 and 140 pixels, respectively.

## Feature Extraction

Feature extraction plays an important role in face recognition. A good feature should be able to distinguish different users. In this part, you will learn to extract PCA feature and LDA feature for faces.

## PCA feature

The main tasks for PCA are:

* Compute the PCA projection matrix We using the training images.
* Project all training images into PCA space, and represents each person (class) by the mean projected vector of his training images.
* Project all testing images into PCA space.

Study the code in FisherFace.py. Use these functions for PCA feature extraction: myPCA,read faces. Their purpose is described in the code. Follow the instructions:

1. \1. Read data.Use read faces to read in all the training images by setting the argument

   to be the path to train folder. You should now have a 22400 by 120 matrix called1

faces, whose columns are all the face images from the train directory, and an array ofcorresponding labels ranging from 0 to 9.

1. **Train PCA.** Using the myPCA function, compute the PCA projection matrix W,the global mean vector me, and the vector of eigenvalues. Read the code in myPCAto understand how this is done. In particular, note the use of the inner product trick(as explained in the lecture notes) to avoid an “Out of memory” error.
2. **Select feature dimension.** Retain only the top K (K = 30) eigenfaces, by typingWe =W[:,:K].
3. **Project training images.** For each vector x in faces, project it into PCA spaceusing: y = WeT (x − me). y is the pca feature representation of x.
4. **Generate templates.** For each person, compute the mean z of all his pca represen-tation vectors. Store these vectors as columns in a Numpy matrix Z. For convenience,it is best to store the columns of Z so that ith column corresponds to person withlabel i. Check that your Z matrix should have the size of K by 10.
5. **Generate templates.** For each person, compute the mean z of all his pca represen-tation vectors. Store these vectors as columns in a Numpy matrix Z. For convenience,it is best to store the columns of Z so that ith column corresponds to person withlabel i. Check that your Z matrix should have the size of K by 10.

## LDA feature

The main tasks for LDA are:

* Compute the LDA projection matrix Wf using the training images.
* Represents each person (class) by the centers of his training images in LDA space. 
* Project all testing image into LDA space.

For the sake of computational cost, PCA is often used to reduce the dimension of inputs beforehand in LDA feature extraction. Use myLDA function in FisherFace.py for LDA feature extraction. In this task, we are actually implementing Whitened FLD in note lecture. Based on the results of PCA projection, follow the instructions bellow:

1. **Dimension reduction.** Reduce the dimension of input (22400) to K1(K1 =90). Retain only the top K1 eigenfaces by W1 = W[:, : K1], where W is the result of step 2 in PCA feature extraction part. Reduce the dimension of x by x′ = W1T (x − me). Apply the dimension reduction to whole input “faces”. You should now have a K by 120 matrix X.


2. **Train LDA.** Apply myLDA on X and its corresponding labels and compute the LDA projection matrix Wf and centers of each class in LDA space C. Read myLDA and understand how this is done.
3. **Generate templates.** C are the templates for all people, with corresponding labels in ClassLabel from myLDA.
4. **Project image.** For a given face x, project it into LDA space by: y = WfT W1T (x − me ).

## Feature fusionIn

some cases, a face recognition method will fuse two features to improve the performance. The fusion is always performed by concatenating two features. Given an image, we are now able to get a PCA feature ye and LDA feature yf . A new feature vector can be constructed by y = [αy   (1 - α)yf]. Set α = 0.5, compute the template of new feature for each person, and represent each test face by the new feature representation.

## Classification

Your classifier is now ready. To classify a new face image x (re-shaped into a vector), firstextract its feature by projecting it into feature space. Then search for the template that isclosest to y, using the Euclidean distance metric. The index of the nearest template revealsthe identity of x. For example, if the nearest template is column 2, the predicted id isclassLabel[1].

You can now evaluate the performance of your classifier. Using the images in the testdirectory, classify each of them with your classifier as explained above.

The Confusion Matrix is a useful way to evaluate the performance a classifier. Each elementcij of an M by M Confusion Matrix C is the number of times an image from person i isclassified as person j by the classifier. Thus the perfect classifier should produce a diagonalConfusion Matrix. Any non-zero off-diagonal element in C represents an error. The overallaccuracy of the classifier can be calculated as the trace divided by the sum of all elements.

Calculate the 10 by 10 confusion matrix as follows. First, initialize all entries in the Con-fusion Matrix to 0. Then, for each test image, let predicted id be the identity that yourclassifier outputs, and let actual id be the actual identity of the test image (which you candetermine from its filename). Add 1 to the entry in the actual id-th row and predicted id-thcolumn of the Confusion Matrix.

## Questions

Use PCA feature, LDA feature, fused feature, and perform the classification respectively. Answer folloing question.

1. Print the confusion matrix and overall accuracy for classifiers with three features.
2. Compare the results for PCA feature and LDA feature, which feature is better? Why?
3. Let α = 0.1, 0.2, . . . , 0.9. Retrain your classifier for fused feature and re-calculate its accuracy for each α. Plot accuracy versus α for different α. Submit this plot. What do you observe?
4. Does the fused feature outperform both PCA feature and LDA feature? Why?