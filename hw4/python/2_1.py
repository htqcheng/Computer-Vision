import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import submission as sub
import cv2
import helper

# Problem 2.1
im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
M = max(im1.shape)
pts = np.load('../data/some_corresp.npz')
pts1 = pts['pts1']
pts2 = pts['pts2']
F = sub.eightpoint(pts1, pts2, M)
np.savez('q2_1', F=F, M=M)
test = np.load('q2_1.npz')
helper.displayEpipolarF(im1, im2, F)

# Problem 3.1
Ks = np.load('../data/intrinsics.npz')
K1 = Ks['K1']
K2 = Ks['K2']
E = sub.essentialMatrix(F, K1, K2)