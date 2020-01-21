import numpy as np
def warp(im, A, output_shape):

	# Pre-allocate empty result image, same shape
	result = np.zeros(output_shape)

	# Get height and width
	h, w = output_shape[0], output_shape[1]
	# h = 200
	# w = 150

	# H = [0, 1, 2, 3, 4, 5 ...... 199]
	H = np.arange(h)
	# W = [0, 1, 2, 3, 4, 5 ...... 149]
	W = np.arange(w)

	# Meshgrid
	WW, HH = np.meshgrid(W, H)
	print(HH)
	print(WW)
	# Intends to create a grid like:
	# [(0,0)   (1,0)  (2,0) ....... (149,0)  ]
	#  .
	#  .
	#  .
	# [(0,199) (1,199) ............ (149,199)]

	# Devided into HH = 
	# [0, 0, 0, 0, 0 .....  0]
	# [1, 1, 1, 1, 1 .....  1]
	#  .
	#  .
	# [199, 199 ......... 199]


	# And WW = 
	# [0, 1, 2, 3 .....   149]
	#  .
	#  .
	# [0, 1, 2, 3 .....   149]

	# HH and WW are 2D arrays with same dimensions
	# If combined together, we will have a map of coordinates

	# Now we flatten both arrays
	ind_Des_Y = HH.flatten()
	ind_Des_X = WW.flatten()
	# ind_Des_X = [0 1 2 3 4 ... 149 0 1 2 ..... 149 ...  0   1  ... 149]
	# ind_Des_Y = [0 0 0 0 0 ...  0  1 1 1 .....  1  ... 149 149 ... 149]

	ind_Des = np.array([ind_Des_Y, ind_Des_X, np.ones(len(ind_Des_X))])
	# Creating a 2D array (30000 x 3), convert into "homogeneous coordinates"
	# [x0 x1 x2 ....... x29999]
	# [y0 y1 y2 ....... y29999]
	# [1  1  1  .......    1  ]

	# If we use match the destination coordinates from the original image with the transformation matrix
	# there is bound to be holes.
	# A * P_original -> P_destination

	# The reason why we allocate an empty destination image is so that we can match every pixel in the destination
	# image, through the "inverse" of the transformation matrix, to the corresponding pixel in the original image,
	# thus ensuring no missed (hole) pixels in the destination image
	# inv_A * P_destination -> P origin
	inv_A = np.linalg.inv(A)
	ind_Orig = np.floor(np.dot(inv_A, ind_Des)).astype(int)
	ind_Orig_Y, ind_Orig_X = ind_Orig[0], ind_Orig[1]
	# This is generate a (30000 x 3) 2D array, with each set of coordinates corresponding to the matching pixels in the original image
	# ind_Des = [0    1    2  3 4 ... 149 0 1 2 ..... 149 ...  0   1  ... 149]
	# 		    [0    0    0  0 0 ...  0  1 1 1 .....  1  ... 149 149 ... 149]
	#           [1    1    1  1 1 ...  1  1 1 1 .....  1  ...  1   1  ...  1 ]

	# ind_Orig= [-56 -55 -53 .........................................       ]
	#           [ 56  55  55 .........................................       ]
	#           [  1   1   1 .........................................       ]
	
	# Mark the cooridates that are invalid in ind_Orig
	# And create an array storing these coordinates
	invalid = np.where(ind_Orig_Y < 0, 1, 0) + np.where(ind_Orig_Y >= h , 1, 0) + np.where(ind_Orig_X < 0, 1, 0) + np.where(ind_Orig_X >= w, 1, 0)
	ind_invalid_Y = np.array([ind_Des_Y[invalid > 0]]) 
	ind_invalid_X = np.array([ind_Des_X[invalid > 0]])

	# clipping: values out of bounds will become the same as the boundary
	# This is to prevent the Destination image pixel matches to a coordinate of the original image that is out of its bounds
	ind_Orig_X = np.clip(ind_Orig_X, 0, w - 1)
	ind_Orig_Y = np.clip(ind_Orig_Y, 0, h - 1)

	# match the pixels
	result[ind_Des_Y, ind_Des_X] = im[ind_Orig_Y, ind_Orig_X]
	# make the pixels on the destination image that were matched to a clipped pixel (out of bounds) black
	result[ind_invalid_Y, ind_invalid_X] = 0



	return result
