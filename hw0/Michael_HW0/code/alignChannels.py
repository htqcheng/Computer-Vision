import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt

def alignChannels(red, green, blue):
	"""Given 3 images corresponding to different channels of a color image,
	compute the best aligned result with minimum abberations

	Args:
		red, green, blue - each is a HxW matrix corresponding to an HxW image

	Returns:
		rgb_output - HxWx3 color image output, aligned as desired"""

	# shift images

	# Green Image
	# Variables for recording the shifts in x, y axis
	x_green = 0
	y_green = 0
	# SSD Score recording
	green_score = sys.maxsize

	# All offsets of the green image
	for i in range(-30, 31):
		for j in range(-30, 31):
			# Roll will shift the values from the end of the matrix to the start, and vice versa
			g = np.roll(green, [i , j], axis = [0, 1])
			# Because the values at the boundaries will be messed up, we will not take the boundary pixels into account
			s = np.sum((red[30 : -30, 30 : -30] - g[30 : -30, 30 : -30]) ** 2)
			# If the SSD score is lower than the current minimum score
			if s < green_score:
				green_score = s
				x_green = i
				y_green = j


	# Blue Image
	x_blue = 0
	y_blue = 0
	blue_score = sys.maxsize

	for i in range(-30, 31):
		for j in range(-30, 31):
			b = np.roll(blue, [i , j], axis = [0, 1])
			s = np.sum((red[30:-30,30:-30] - b[30:-30,30:-30]) ** 2)
			if s < blue_score:
				blue_score = s
				x_blue = i
				y_blue = j


	# Combining the 3 channels 
	I = len(red)
	J = len(red[0])
	rgbArray = np.zeros([I, J, 3], dtype = np.uint8)
	rgbArray[..., 0] = red 
	rgbArray[..., 1] = np.roll(green, [x_green , y_green], axis = [0, 1])
	rgbArray[..., 2] = np.roll(blue, [x_blue , y_blue], axis = [0, 1])

	img = Image.fromarray(rgbArray)


	
	return img


