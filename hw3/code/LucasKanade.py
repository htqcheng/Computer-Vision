import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Initialize warp matrix p
    p = p0
    # Assume input images don't need to be processed (RGB2Grey or similar)
    # note numr is y and numc is x for cartesian
    # points = []
    # numr = int(rect[3]-rect[1]+1)
    # numc = int(rect[2]-rect[0]+1)
    # number of points
    # N = numr*numc
    # for i in range(numc):
    #     for j in range(numr):
    #         points.append([int(rect[0]+i), int(rect[1]+j)])
    # # points become a 2 by N matrix, note it is x, y
    # points = np.asarray(points).T
    # append 1s to the bottom
    # points = np.vstack(points, np.ones((points.shape[1], 1)))
    # should be 2xN
    # print("The shape of points should be 2xN: " + str(points.shape))
    # get original intensities, shift image to estimate partial rect positions
    # x_shift = rect[0]%1
    # y_shift = rect[1]%1
    # shifted_It = ndimage.shift(It, [-y_shift, -x_shift])
    # T_x = shifted_It[points[1, :], points[0, :]]
    xs = np.arange(rect[0], rect[2]+1, 1)
    ys = np.arange(rect[1], rect[3]+1, 1)
    It_x = np.arange(0, It.shape[1], 1)
    It_y = np.arange(0, It.shape[0], 1)
    Rect_It = RectBivariateSpline(It_y, It_x, It)
    T_x = Rect_It(ys, xs)
    N = T_x.size
    print("The shape of T(x) is : " + str(T_x.shape))

    #iterate over to find true p
    for i in range(int(num_iters)):
        # account for fractional location after warp
        shifted_It1 = ndimage.shift(It1, [-p[1], -p[0]])
        # get warped image 1D vector
        Rect_It1 = RectBivariateSpline(It_y, It_x, shifted_It1)
        It1_patch = Rect_It1(ys, xs)
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
        Rect_x = RectBivariateSpline(It_y, It_x, image_grad_x)
        Rect_y = RectBivariateSpline(It_y, It_x, image_grad_y)
        grad_x_patch = Rect_x(ys, xs)
        grad_y_patch = Rect_y(ys, xs)
        A_T = np.zeros((2, N))
        # A_T[0, :] = grad_x_patch.reshape(-1)
        # A_T[1, :] = grad_y_patch.reshape(-1)
        # ravel is faster
        A_T[0, :] = grad_x_patch.ravel()
        A_T[1, :] = grad_y_patch.ravel()
        hessian = A_T @ A_T.T
        # print("The shape of hessian is: " + str(hessian.shape))
        delta_p = np.linalg.inv(hessian) @ A_T @ b
        # print(np.linalg.norm(delta_p))
        p[0] += delta_p[0]
        p[1] += delta_p[1]
        # print(p)
        if np.linalg.norm(delta_p) < threshold:
            break

    return p
