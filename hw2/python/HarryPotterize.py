import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
import matplotlib.pyplot as plt


#Write script for Q2.2.4
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# Change colors from OpenCV's BGR to RGB and resize harry potter to be the same as cv_cover
hp_cover = cv2.cvtColor(hp_cover, cv2.COLOR_BGR2RGB)
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
cv_desk = cv2.cvtColor(cv_desk, cv2.COLOR_BGR2RGB)

matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)
# N = matches.shape[0]
# locs1_matched = np.zeros((N, 2))
# locs2_matched = np.zeros((N, 2))
# for i in range(N):
#     locs1_matched[i] = locs1[matches[i, 0]]
#     locs2_matched[i] = locs2[matches[i, 1]]
locs1_matched = locs1[matches[:, 0]]
locs2_matched = locs2[matches[:, 1]]
H, inliers = computeH_ransac(locs1_matched, locs2_matched, opts)

# h, w, _ = cv_desk.shape
# hp_warped = cv2.warpPerspective(hp_cover, H, (w, h))
# plt.imshow(hp_warped)
# plt.show()

composite_img = compositeH(H, cv_desk, hp_cover)
plt.imshow(composite_img)
plt.show()
