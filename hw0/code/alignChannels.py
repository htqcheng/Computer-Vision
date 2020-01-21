import numpy as np


def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    red_n = red[30:-40, 30:-40]
    red_n = red_n.astype(int)
    g_off = np.zeros((61, 61))
    b_off = np.zeros((61, 61))
    for i in range(-30, 31, 1):
        for j in range(-30, 31, 1):
            blue_n = blue[30+i:-40+i, 30+j:-40+j]
            blue_n = blue_n.astype(int)
            green_n = green[30+i:-40+i, 30+j:-40+j]
            green_n = green_n.astype(int)
            g_score = np.sum((red_n-green_n)**2)
            b_score = np.sum((red_n-blue_n)**2)
            g_off[i+30, j+30] = g_score
            b_off[i+30, j+30] = b_score

    green_offset = np.unravel_index(np.argmin(g_off), g_off.shape)
    blue_offset = np.unravel_index(np.argmin(b_off), b_off.shape)
    g_i = green_offset[0]
    g_j = green_offset[1]
    b_i = blue_offset[0]
    b_j = blue_offset[1]
    print(g_i, g_j, b_i, b_j)
    r_blank = np.zeros((870, 1003), dtype=np.uint8)
    b_blank = np.zeros((870, 1003), dtype=np.uint8)
    g_blank = np.zeros((870, 1003), dtype=np.uint8)
    for i in range(red.shape[0]):
        for j in range(red.shape[1]):
            r_blank[i+30, j+30] = red[i, j]
            g_blank[i+g_i, j+g_j] = green[i, j]
            b_blank[i+b_i, j+b_j] = blue[i, j]

    red = r_blank
    green = g_blank
    blue = b_blank

    t = np.stack((red, green, blue), axis=2)
    print(t.shape)

    return t
