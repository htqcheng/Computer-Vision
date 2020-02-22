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
	eig_values, eig_vectors = np.linalg.eig(np.transpose(A)@A)
	min_vec = eig_vectors[:, np.argmin(eig_values)]
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
	max_x1 = max_x2 = -1
	# for i in range(num_points):
	# 	x1_dist = np.sqrt((x1_[i, 0])**2 + (x1_[i, 1])**2)
	# 	x2_dist = np.sqrt((x2_[i, 0])**2 + (x2_[i, 1])**2)
	# 	max_x1 = max(max_x1, x1_dist)
	# 	max_x2 = max(max_x2, x2_dist)
	x1_dist = np.sqrt(x1_[:, 0]**2 + x1_[:, 1]**2)
	x2_dist = np.sqrt(x2_[:, 0]**2 + x2_[:, 1]**2)
	max_x1 = np.max(x1_dist)
	max_x2 = np.max(x2_dist)
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
	x1_scale = np.eye(3)*x1_norm
	x1_scale[2, 2] = 1
	x2_scale = np.eye(3)*x2_norm
	x2_scale[2, 2] = 1
	T1 = x1_scale @ x1_trans
	# print(T1)
	T2 = x2_scale @ x2_trans
	# print(T2)

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
	locs1 = np.flip(locs1, axis=1)
	locs2 = np.flip(locs2, axis=1)
	N = locs1.shape[0]
	inliers = np.zeros(N)
	bestH2to1 = []
	for i in range(max_iters):
		rand_pts = np.random.choice(N, 4, replace=False)
		H2to1 = computeH_norm(locs1[rand_pts], locs2[rand_pts])
		# print(H2to1)
		inlier = np.zeros(N)
		# do without for loop
		locs2_p = np.concatenate((locs2.T, np.ones((1, N))), axis=0)
		projected = H2to1 @ locs2_p
		non_homogeneous = np.zeros((N, 2))
		non_homogeneous[:, 0] = projected[0, :]/projected[2, :]
		non_homogeneous[:, 1] = projected[1, :]/projected[2, :]
		diff = locs1 - non_homogeneous
		normalized = np.sqrt(np.sum(diff**2, axis=1))
		inlier = normalized <= inlier_tol
		## for loop method
		# for p in range(N):
		# 	locs2_p = np.concatenate((locs2[p], [1]))
		# 	projected = H2to1 @ locs2_p
		# 	non_homogeneous = np.array([projected[0]/projected[2], projected[1]/projected[2]])
		# 	diff = locs1[p] - non_homogeneous
		# 	normalized = np.sqrt(sum(diff**2))
		# 	# print(normalized)
		# 	if normalized <= inlier_tol:
		# 		inlier[p] = 1
		# print(sum(inlier))
		if sum(inlier) > sum(inliers):
			inliers = inlier
			bestH2to1 = H2to1
	# if (bestH2to1 == []):
	# 	bestH2to1 = H2to1

	return bestH2to1, inliers



def compositeH(H2to1, template, img):

	#Create mask of same size as image
	mask = np.ones(img.shape, dtype=np.uint8)

	#Warp mask by appropriate homography
	h, w, _ = template.shape
	warped_mask = cv2.warpPerspective(mask, H2to1, (w, h))
	#Warp template by appropriate homography
	warped_img = cv2.warpPerspective(img, H2to1, (w, h))
	#Use mask to combine the warped template and the image
	composite_img = template*(1-warped_mask) + warped_img*warped_mask
	
	return composite_img


