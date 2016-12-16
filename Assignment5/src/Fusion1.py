import hw5.FisherFace as ff
import hw5.PCA1 as pca1
import hw5.LDA1 as lda1
import numpy as np

template_pca = pca1.z_train
template_pca = np.matrix(template_pca)
template_lda = lda1.C
template_lda = np.matrix(template_lda)
a = 0.7
template_fusion = np.concatenate((np.multiply(template_pca, a), np.multiply(template_lda, (1 - a))))
template_fusion.shape

faces_test = pca1.faces_test
faces_test = np.matrix(faces_test)

def transferToMatrix(m):
    return np.matrix(m)

W_et = pca1.W_et
W_et = transferToMatrix(W_et)
Wf_t = lda1.Wf_t
Wf_t = transferToMatrix(Wf_t)
W1_t = lda1.W1_t
W1_t = transferToMatrix(W1_t)
m = pca1.m
m = transferToMatrix(m)

def testImageVectors(j):
    vector_pca = np.dot(W_et, (faces_test[:, j] - m.transpose()))
    vector_lda = np.dot(np.dot(Wf_t, W1_t), (faces_test[:, j] - m.transpose()))
    return np.concatenate((np.multiply(vector_pca, a), np.multiply(vector_lda, (1 - a))))

z_test = np.zeros((39, 10))
z_test = np.matrix(z_test)

confusion_matrix = np.zeros((10, 10))
for i in range(0, 120):
    for l in range(0, 10):
        z_test[:, l] = testImageVectors(i)
    z_test = np.subtract(z_test, template_fusion)
    r = ff.ComputeNorm(z_test)
    index_y = r.argmin()
    index_x = int(i / 12)
    confusion_matrix[index_x][index_y] += 1

print(confusion_matrix)

accuracy = np.trace(confusion_matrix) / 120
print("Accuracy(np.trace(confusion_matrix) / 120): " + str(accuracy))