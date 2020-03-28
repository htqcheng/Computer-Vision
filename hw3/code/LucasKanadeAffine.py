import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    y_len, x_len = It1.shape
    N = It1.size
    # print("The shape of T(x) is : " + str(T_x.shape))

    image_grad_y = np.gradient(It1, axis=0)
    image_grad_x = np.gradient(It1, axis=1)

    # iterate over to find true p
    for i in range(int(num_iters)):
        # shift It1 by M to compare with It
        # Not sure if inverse M is correct here
        shifted_It1 = ndimage.affine_transform(It1, M)
        # get mask for It
        It_mask = np.ones(It1.shape)
        It_mask = ndimage.affine_transform(It_mask, M)
        # get warped image 1D vector
        T_x = It * It_mask
        # print("The shape of T(x) is: " + str(T_x.shape))
        # print("The shape of It is: " + str(It.shape))
        # computer error b (1xN)
        b = T_x - shifted_It1
        b = b.reshape(-1)
        # print("The shape of b is: " + str(b.shape))
        # reshape and compute gradients
        warp_grad_y = ndimage.affine_transform(image_grad_y, M).ravel()
        warp_grad_x = ndimage.affine_transform(image_grad_x, M).ravel()
        grad_I = np.zeros((N, 2))
        grad_I[:, 0] = warp_grad_x
        grad_I[:, 1] = warp_grad_y
        A = np.zeros((N, 6))
        # Assume the x's get reshaped to a row first
        for c in range(N):
            y = c // x_len
            x = c % x_len
            # print(grad_I[c, :])
            dWdP = np.array([[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]])
            # dWdP = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
            A[c, :] = grad_I[c, :] @ dWdP
        hessian = A.T @ A
        # print("The shape of hessian is: " + str(hessian.shape))
        delta_p = np.linalg.inv(hessian) @ A.T @ b

        # print("The shape of delta_p is: " + str(delta_p.shape))
        # M += delta_p.reshape((2, 3))
        # print(np.linalg.norm(delta_p))
        M[0,0] += delta_p[4]
        M[0,1] += delta_p[3]
        M[0,2] += delta_p[5]
        M[1,0] += delta_p[1]
        M[1,1] += delta_p[0]
        M[1,2] += delta_p[2]

        # print(delta_p)
        if np.linalg.norm(delta_p) < threshold:
            print("success")
            break
    # M = M[0:2, :]
    return M
