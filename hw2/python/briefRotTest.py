import numpy as np
import cv2
from matchPics import matchPics
import scipy.ndimage
from helper import plotMatches
from opts import get_opts
import matplotlib.pyplot as plt

#Q2.1.6
#Read the image and convert to grayscale, if necessary
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')

histogram = np.zeros(36)
angles = np.zeros(36)
for i in range(36):
    # Rotate Image
    rotated = scipy.ndimage.rotate(cv_cover, i*10)
    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, rotated, opts)
    #Update histogram
    histogram[i] = matches.shape[0]
    angles[i] = 10*i
    if i == 1 or i == 9 or i == 18:
        plotMatches(cv_cover, rotated, matches, locs1, locs2)
    print(histogram)
    print(i)

#Display histogram
plt.bar(angles, histogram, width=10, align='center')
plt.xlabel('Rotation Angle')
plt.ylabel('Number of Matches')
plt.title('BRIEF and Rotation')
plt.show()
