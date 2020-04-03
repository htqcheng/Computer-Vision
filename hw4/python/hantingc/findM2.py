'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import submission as sub
import helper

pts = np.load('../data/some_corresp.npz')
pts1 = pts['pts1']
pts2 = pts['pts2']
q2_1 = np.load('q2_1.npz')
F = q2_1['F']

Ks = np.load('../data/intrinsics.npz')
K1 = Ks['K1']
K2 = Ks['K2']
E = sub.essentialMatrix(F, K1, K2)

M2s = helper.camera2(E)
M1 = np.hstack((np.eye(3), np.zeros((3,1))))
# print(M2s.shape[2])
for i in range(M2s.shape[2]):
    M2 = M2s[:,:,i]
    C1 = K1@M1
    C2 = K2@M2
    P = sub.triangulate(C1, pts1, C2, pts2)
    # print(i)
    # print(P[:, 2], '\n')
    # print('\n')
    if (P[:, 2] > 0).all():
        print(P[:, 2], '\n')
        break

np.savez('q3_3', M2=M2, C2=C2, P=P)