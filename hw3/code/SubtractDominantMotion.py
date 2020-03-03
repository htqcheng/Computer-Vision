import numpy as np
from scipy import ndimage
from LucasKanadeAffine import LucasKanadeAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # edges where image1 and 2 don't both show image could be a problem
    mask = np.ones(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    image1_affine = ndimage.affine_transform(image1, M)
    diff = image2 - image1_affine
    mask = diff>tolerance

    return mask
