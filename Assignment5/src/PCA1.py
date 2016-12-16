import hw5.FisherFace as ff
import numpy as np

faces_train, idLabel_train = ff.read_faces("train")
W, LL, m = ff.myPCA(faces_train)
W = np.matrix(W)
faces_train = np.matrix(faces_train)
m = np.matrix(m)

K = 30
W_e = W[: ,: K]
W_et = np.transpose(W_e)

z_train = np.zeros((30, 10))
z_train = np.matrix(z_train)

for i in range(0, 10):
    for j in range(0, 12):
        z_train[:, i] += np.dot(W_et, (faces_train[:, i * 12 + j] - m.transpose()))
    z_train[:, i] /= 12

faces_test, idLabel_test = ff.read_faces("test")
faces_test = np.matrix(faces_test)

z_test = np.zeros((30, 10))
z_test = np.matrix(z_test)

def testImageVectors(j):
    return np.dot(W_et, (faces_test[:, j] - m.transpose()))

confusion_matrix = np.zeros((10, 10))

for i in range(0, 120):
    for l in range(0, 10):
        z_test[:, l] = testImageVectors(i)
    z_test = np.subtract(z_test, z_train)
    r = ff.ComputeNorm(z_test)
    index_y = r.argmin()
    index_x = int(i / 12)
    confusion_matrix[index_x][index_y] += 1

confusion_matrix

accuracy = np.trace(confusion_matrix) / 120
print("Accuracy(sum / 120): " + str(accuracy))