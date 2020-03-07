import numpy as np
from scipy import ndimage
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

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
    # mask = np.ones(image1.shape, dtype=bool)
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    image2_affine = ndimage.affine_transform(image2, M)
    image1_mask = np.ones(image1.shape)
    image1_mask = ndimage.affine_transform(image1_mask, M)
    diff = image1*image1_mask - image2_affine
    mask = abs(diff) > tolerance
    # mask = ndimage.morphology.binary_erosion(mask, iterations=1)
    mask = ndimage.morphology.binary_dilation(mask, iterations=5)
    mask = ndimage.morphology.binary_erosion(mask, iterations=5)

    return mask
