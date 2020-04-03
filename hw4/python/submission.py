"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import scipy

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
    # pts1[:,0] = pts1[:,0]*2/M-1
    # pts1[:,1] = pts1[:,1]*2/M-1
    # pts2[:,0] = pts2[:,0]*2/M-1
    # pts2[:,1] = pts2[:,1]*2/M-1

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
    min_vec = min_vec/min_vec[-1]
    F = min_vec.reshape((3, 3))
    # Enforce singularity condition
    F_sing = helper.refineF(F, pts1_norm[:,0:2], pts2_norm[:,0:2])
    # w, diag, vt = np.linalg.svd(F)
    # diag[2] = 0
    # F_sing = w@np.diagflat(diag)@vt
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
    P = np.zeros((N,3))
    err = 0
    for i in range(N):
        A = np.zeros((4,4))
        A[0, :] = C1[0,:] - (C1[2,:]*pts1[i,0])
        A[1, :] = C1[1,:] - (C1[2,:]*pts1[i,1])
        A[2, :] = C2[0,:] - (C2[2,:]*pts2[i,0])
        A[3, :] = C2[1,:] - (C2[2,:]*pts2[i,1])
        # Compute wi
        eig_values, eig_vectors = np.linalg.eig((A.T) @ A)
        wi = eig_vectors[:, np.argmin(eig_values)]
        w = np.zeros(3)
        w[0] = wi[0]/wi[3]
        w[1] = wi[1]/wi[3]
        w[2] = wi[2]/wi[3]
        P[i] = w
        # Update error for inspection
        x1_h = C1@wi
        x2_h = C2@wi
        x1 = np.array([x1_h[0]/x1_h[2], x1_h[1]/x1_h[2]])
        x2 = np.array([x2_h[0]/x2_h[2], x2_h[1]/x2_h[2]])
        err += np.linalg.norm(pts1[i] - x1) + np.linalg.norm(pts2[i] - x2)

    print(err)
    return P


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
    print('p1: ', x1, ', ', y1)
    point = np.array([x1, y1, 1])
    window = 15
    length = 80
    line = F@point
    ssd = np.zeros(length)
    for c in range(length):
        y2 = int(y1-length/2+c)
        x2 = int(round((-line[2]-y2*line[1])/line[0]))
        p = np.array([x2, y2, 1])
        I1 = im1[y1-window:y1+window, x1-window:x1+window]
        I2 = im2[y2-window:y2+window, x2-window:x2+window]
        ssd[c] = sum(sum(sum((I1-I2)**2)))
    c = np.argmin(ssd)
    y2 = int(y1-length/2+c)
    x2 = int(round((-line[2]-y2*line[1])/line[0]))
    print('p2: ', x2, ', ', y2)
    return x2, y2

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
    N = pts1.shape[0]
    print(N)
    # pts1[:, 0] = pts1[:, 0] * 2 / M - 1
    # pts1[:, 1] = pts1[:, 1] * 2 / M - 1
    # pts2[:, 0] = pts2[:, 0] * 2 / M - 1
    # pts2[:, 1] = pts2[:, 1] * 2 / M - 1
    pts1_h = np.hstack((pts1, np.ones((N, 1)))).T
    pts2_h = np.hstack((pts2, np.ones((N, 1)))).T
    max_inlier = 0
    F_out = None
    inliers = None
    for i in range(nIters):
        rand_pts = np.random.choice(N, 8, replace=False)
        F = eightpoint(pts1[rand_pts], pts2[rand_pts], M)
        dist = abs(np.sum(pts2_h*(F@pts1_h), axis=0))
        # print(dist)
        inlier_num = np.sum((dist < tol))
        print(inlier_num)
        if inlier_num>max_inlier:
            max_inlier = inlier_num
            inliers = (dist<tol).reshape(-1)
            F_out = F

    return F_out, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.sqrt(np.sum(r**2))
    if theta == 0:
        ax = r
    else:
        ax = r/theta
    ax_c = np.array([[0, -ax[2, 0], ax[1, 0]], \
                     [ax[2, 0], 0, -ax[0, 0]], \
                     [-ax[1, 0], ax[0, 0], 0]])
    ax_c_square = np.dot(ax, ax.T) - np.sum(ax**2) * np.eye(3)
    R = np.sin(theta) * ax_c + (1 - np.cos(theta)) * ax_c_square + np.eye(3)

    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    ux = (R - R.T)
    u = np.array([ux[2, 1], ux[0, 2], ux[1, 0]])
    s = np.linalg.norm(u)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2

    if s == 0 and c == -1.:
        temp = R + np.diag(np.array([1, 1, 1]))
        v = None
        for i in range(3):
            if np.sum(temp[:, i]) != 0:
                v = temp[:, i]
                break
        temp2 = v / np.sqrt(np.sum(v ** 2))
        r = np.reshape(temp2 * np.pi, (3, 1))
        if np.sqrt(np.sum(r**2)) == np.pi and \
                ((r[0, 0] == 0. and r[1, 0] == 0. and r[2, 0] < 0) or \
                 (r[0, 0] == 0. and r[1, 0] < 0) or (r[0, 0] < 0)):
            return -r
        return r

    if s == 0 and c == 1:
        r = np.zeros((3, 1))
        return r
    else:
        u = u/s
        theta = np.arctan2(s, c)
        r = u * theta
        return r

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
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    x3 = pts3[:, 0]
    y3 = pts3[:, 1]

    N = pts1.shape[0]
    P = np.zeros((N,3))
    err = 0
    for i in range(N):
        conf = np.array([pts1[i, 2], pts2[i, 2], pts3[i, 2]]) > Thres
        if sum(conf) < 2:
            continue
        elif sum(conf) == 2:
            A = np.zeros((4,4))
            A[0, :] = C1[0, :] - (C1[2, :] * pts1[i, 0])
            A[1, :] = C1[1, :] - (C1[2, :] * pts1[i, 1])
            A[2, :] = C2[0, :] - (C2[2, :] * pts2[i, 0])
            A[3, :] = C2[1, :] - (C2[2, :] * pts2[i, 1])
        else:
            A = np.zeros((6,4))
            A[0, :] = C1[0, :] - (C1[2, :] * pts1[i, 0])
            A[1, :] = C1[1, :] - (C1[2, :] * pts1[i, 1])
            A[2, :] = C2[0, :] - (C2[2, :] * pts2[i, 0])
            A[3, :] = C2[1, :] - (C2[2, :] * pts2[i, 1])
            A[4, :] = C3[0, :] - (C3[2, :] * pts3[i, 0])
            A[5, :] = C3[1, :] - (C3[2, :] * pts3[i, 1])

        # Compute wi
        eig_values, eig_vectors = np.linalg.eig((A.T) @ A)
        wi = eig_vectors[:, np.argmin(eig_values)]
        w = np.zeros(3)
        w[0] = wi[0] / wi[3]
        w[1] = wi[1] / wi[3]
        w[2] = wi[2] / wi[3]
        P[i] = w
        # Update error for inspection
        x1_h = C1 @ wi
        x2_h = C2 @ wi
        x1 = np.array([x1_h[0] / x1_h[2], x1_h[1] / x1_h[2]])
        x2 = np.array([x2_h[0] / x2_h[2], x2_h[1] / x2_h[2]])
        if sum(conf) > 2:
            x3_h = C3 @ wi
            x3 = np.array([x3_h[0] / x3_h[2], x3_h[1] / x3_h[2]])
            err += np.linalg.norm(pts3[i, :2] - x3)

        err += np.linalg.norm(pts1[i, :2] - x1) + np.linalg.norm(pts2[i, :2] - x2)

    print(err)
    return P, err
