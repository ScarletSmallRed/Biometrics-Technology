import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator

P = np.array([[3], [3], [1]], dtype = float)
T1 = np.array([[1, 0, -2],[0, 1, -2], [0, 0, 1]], dtype = float)
T2 = np.array([[1, 0, 2],[0, 1, 2],[0, 0, 1]], dtype = float)
ax=[0, 0, 0, 0, 0, 0, 0, 0]
# Create Chart1
plt.figure(1)

for k in range(0, 8):
    # Create sub-chart in Chart
    ax[k]=plt.subplot(2, 4, k+1)

plt.figure(1)
plt.plot(3, 3, 'b', marker = '.')
plt.plot(2, 2, 'g', marker = '.')
plt.title('Handsome P')
plt.sca(ax[0])
plt.xlim(0, 6)
plt.ylim(0, 6)

for i in range(1, 8):
    a = np.pi * i / 4

    R=np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype = float)

    C1 = np.dot(T1, P)
    C2 = np.dot(R, C1)
    C3 = np.dot(T2,C2)

    plt.plot(C3[0][0], C3[1][0], 'r', marker = '.')
    plt.plot(2, 2, 'c', marker = '.')
    plt.title("Handsome Rotate " + str(i) + "pi/4")
    plt.sca(ax[i])
    plt.xlim(0, 5)
    plt.ylim(0, 5)

plt.show()
    



