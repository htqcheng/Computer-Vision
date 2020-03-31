"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[2/M, 0, -1], [0, 2/M, -1], [0, 0, 1]])
    N = pts1.shape[0]
    pts1_h = np.hstack((pts1, np.ones((N,1)))).T
    pts2_h = np.hstack((pts2, np.ones((N,1)))).T
    pts1_norm = (T@pts1_h).T
    pts2_norm = (T@pts2_h).T

    # Build the matrix U
    U = np.zeros((N,9))
    U[:,0] = pts1_norm[:,0]*pts2_norm[:,0]
    U[:,1] = pts1_norm[:,0]*pts2_norm[:,1]
    U[:,2] = pts1_norm[:,0]
    U[:,3] = pts1_norm[:,1]*pts2_norm[:,0]
    U[:,4] = pts1_norm[:,1]*pts2_norm[:,1]
    U[:,5] = pts1_norm[:,1]
    U[:,6] = pts2_norm[:,0]
    U[:,7] = pts2_norm[:,1]
    U[:,8] = np.ones((N))
    # Compute F
    eig_values, eig_vectors = np.linalg.eig((U.T)@U)
    min_vec = eig_vectors[:, np.argmin(eig_values)]
    F = min_vec.reshape((3, 3))
    # Enforce singularity condition
    w, diag, vt = np.linalg.svd(F)
    diag[2] = 0
    F_sing = w@np.diagflat(diag)@vt
    # F_sing = refineF(F_sing, pts1, pts2)
    # unnormalize F
    F_unnorm = T.T@F_sing@T

    return F_unnorm


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K1.T@F@K2
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    N = pts1.shape[0]
    for i in range(N):
        A = np.zeros((4,4))
        A[0, :] = C1[0,:] - (C1[2,:]*pts1[i,0])
        A[1, :] = C1[1,:] - (C1[2,:]*pts1[i,1])
        A[2, :] = C2[0,:] - (C2[2,:]*pts2[i,0])
        A[3, :] = C2[1,:] - (C2[2,:]*pts2[i,1])



'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters, tol):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation
    pass
