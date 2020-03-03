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

    T_x = It[20:-20, 20:-20]  # cut out parts of the image so all is contained in It1
    # xs = np.arange(rect[0], rect[2] + 1, 1)
    # ys = np.arange(rect[1], rect[3] + 1, 1)
    # It_x = np.arange(0, It.shape[1], 1)
    # It_y = np.arange(0, It.shape[0], 1)
    # Rect_It = RectBivariateSpline(It_y, It_x, It)
    # T_x = Rect_It(ys, xs)
    y_len, x_len = T_x.shape
    N = T_x.size
    print("The shape of T(x) is : " + str(T_x.shape))

    # iterate over to find true p
    for i in range(int(num_iters)):
        # account for fractional location after warp
        shifted_It1 = ndimage.affine_transform(It1, np.linalg.inv(M))
        # get warped image 1D vector
        It1_patch = shifted_It1[20:-20, 20:-20]
        # computer error b (1xN)
        b = T_x - It1_patch
        b = b.reshape(-1)
        # print("The shape of b is: " + str(b.shape))
        # reshape and compute gradients. Not sure if sobel axis is correct
        # maybe np.gradient
        image_grad_y = np.gradient(shifted_It1, axis=0)
        image_grad_x = np.gradient(shifted_It1, axis=1)
        # image_grad_y = ndimage.sobel(shifted_It1, axis=0)
        # image_grad_x = ndimage.sobel(shifted_It1, axis=1)
        grad_x_patch = image_grad_x[20:-20, 20:-20].reshape((N, 1))
        grad_y_patch = image_grad_y[20:-20, 20:-20].reshape((N, 1))
        grad_I = np.zeros((N, 2))
        grad_I[:, 0] = grad_x_patch
        grad_I[:, 1] = grad_y_patch
        A = np.zeros((N, 6))
        for c in range(N):
            y = 20 + c//x_len
            x = 20 + c % x_len
            dWdP = np.array([[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]])
            A[c, :] = grad_I[c, :] @ dWdP
        hessian = A.T @ A
        # print("The shape of hessian is: " + str(hessian.shape))
        delta_p = np.linalg.inv(hessian) @ A.T @ b

        print(delta_p.shape)
        M += delta_p.reshape((2, 3))
        # print(np.linalg.norm(delta_p))

        # print(p)
        if np.linalg.norm(delta_p) < threshold:
            break

    return M
