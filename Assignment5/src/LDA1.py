import hw5.FisherFace as ff
import numpy as np

faces_train, idLabel_train = ff.read_faces("train")
W, LL, m = ff.myPCA(faces_train)
W = np.matrix(W)
faces_train = np.matrix(faces_train)
m = np.matrix(m)

K1 = 90
W1 = W[: ,: K1]
W1_t = np.transpose(W1)

X = np.zeros((K1, 120))
X = np.matrix(X)

for i in range(0, 120):
    X[: ,i] = np.dot(W1_t, (faces_train[: ,i] - m.transpose()))
X = np.asarray(X)

Wf, C, classLabels = ff.myLDA(X, idLabel_train)
Wf = np.matrix(Wf)
Wf_t = np.transpose(Wf)
C = np.matrix(C)

faces_test, idLabel_test = ff.read_faces("test")
faces_test = np.matrix(faces_test)


def testImageVectors(j):
    return np.dot(np.dot(Wf_t, W1_t), (faces_test[:, j] - m.transpose()))


z_test = np.zeros((9, 10))
z_test = np.matrix(z_test)

confusion_matrix = np.zeros((10, 10))

for i in range(0, 120):
    for l in range(0, 10):
        z_test[:, l] = testImageVectors(i)
    z_test = np.subtract(z_test, C)
    r = ff.ComputeNorm(z_test)
    index_y = r.argmin()
    index_x = int(i / 12)
    confusion_matrix[index_x][index_y] += 1

confusion_matrix

accuracy = np.trace(confusion_matrix) / 120
print("Accuracy(sum / 120): " + str(accuracy))