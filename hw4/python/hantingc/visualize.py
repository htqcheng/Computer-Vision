'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import submission as sub
import matplotlib.pyplot as plt

img1 = plt.imread('../data/im1.png')
img2 = plt.imread('../data/im2.png')
q2_1 = np.load('q2_1.npz')
F = q2_1['F']

# Get the handpicked points
coord = np.load('../data/templeCoords.npz')
x1 = coord['x1']
y1 = coord['y1']
# Get the Ks
Ks = np.load('../data/intrinsics.npz')
K1 = Ks['K1']
K2 = Ks['K2']
# Get the Cs
q3_3 = np.load('q3_3.npz')
M2 = q3_3['M2']
C2 = q3_3['C2']
M1 = np.hstack((np.eye(3), np.zeros((3,1))))
C1 = K1@M1

# Use Epipolar correspondence to find x2 and y2
N = len(x1)
pts1 = np.array([x1.reshape(-1), y1.reshape(-1)]).T
print(pts1.shape)
pts2 = np.zeros((N,2))
for i in range(N):
    x2, y2 = sub.epipolarCorrespondence(img1, img2, F, int(x1[i]), int(y1[i]))
    pts2[i] = [x2, y2]

# Now use triangulate to find 3D points
P = sub.triangulate(C1, pts1, C2, pts2)
np.savez('q4_2.npz', F = F, M1 = M1, M2 = M2, C1=C1, C2=C2)
print(P)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b')
ax.set_ylim3d(-0.7,0.7)
plt.show()