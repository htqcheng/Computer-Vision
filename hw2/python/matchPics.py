import numpy as np
import cv2
import skimage.color
from skimage.color import rgb2gray
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches


def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

	#Convert Images to GrayScale
	I1g = rgb2gray(I1)
	I2g = rgb2gray(I2)
	
	#Detect Features in Both Images
	locs1_ = corner_detection(I1g, sigma)
	locs2_ = corner_detection(I2g, sigma)
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1g, locs1_)
	desc2, locs2 = computeBrief(I2g, locs2_)

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio)
	# print(matches)

	return matches, locs1, locs2

