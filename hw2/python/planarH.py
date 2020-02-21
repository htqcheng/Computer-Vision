import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	A = np.zeros((8, 9))
	for i in range(x1.shape[0]):
		A[0+2*i] = [x2[i, 0], x2[i, 1], 1, 0, 0, 0, -x1[i][0]*x2[i, 0], -x1[i][0]*x2[i, 1], -x1[i][0]]
		A[1+2*i] = [0, 0, 0, x2[i, 0], x2[i, 1], 1, -x1[i][1]*x2[i, 0], -x1[i][1]*x2[i, 1], -x1[i][1]]
	# find eigenvalues of A'A and pick the smallest
	eig_values, eig_vectors = np.inalg.eig(A)
	min_val = np.min(eig_values)
	min_vec = eig_vectors[np.argmin(eig_values)]
	H2to1 = min_vec.reshape((3, 3))

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	num_points = x1.shape[0]
	[x1_centroid, y1_centroid] = np.sum(x1, axis=0)/num_points
	[x2_centroid, y2_centroid] = np.sum(x1, axis=0)/num_points

	#Shift the origin of the points to the centroid
	x1_ = np.zeros((x1.shape))
	x1_[:, 0] = x1[:, 0] - x1_centroid
	x1_[:, 1] = x1[:, 1] - y1_centroid
	x2_ = np.zeros((x2.shape))
	x2_[:, 0] = x2[:, 0] - x2_centroid
	x2_[:, 1] = x2[:, 1] - y2_centroid

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max_x1 = max_x2 = 0
	for i in range(num_points):
		x1_dist = np.sqrt((x1_[i, 0])**2 + (x1_[i, 1])**2)
		x2_dist = np.sqrt((x2_[i, 0])**2 + (x2_[i, 1])**2)
		max_x1 = max(max_x1, x1_dist)
		max_x2 = max(max_x2, x2_dist)
	x1_norm = np.sqrt(2)/max_x1
	x2_norm = np.sqrt(2)/max_x2
	x1_ = x1_*x1_norm
	x2_ = x2_*x2_norm

	#Similarity transform 1
	x1_trans = np.eye(3)
	x1_trans[0, 2] = -x1_centroid
	x1_trans[1, 2] = -y1_centroid
	x2_trans = np.eye(3)
	x2_trans[0, 2] = -x2_centroid
	x2_trans[1, 2] = -y2_centroid

	#Similarity transform 2
	x1_scale = np.eye(2)*x1_norm
	x2_scale = np.eye(2)*x2_norm
	T1 = x1_scale @ x1_trans
	print(T1)
	print(T1 @ x1 == x1_)
	T2 = x2_scale @ x2_trans
	print(T2)
	print(T2 @ x2 == x2_)

	#Compute homography
	H = computeH(x1_, x2_)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H @ T2

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	


	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	
	return composite_img


