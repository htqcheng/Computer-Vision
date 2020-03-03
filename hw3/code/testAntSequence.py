import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

# seq = np.load('../data/antseq.npy')
seq = np.load('../../../CV_Large_data/hw3/data/antseq.npy')
cuts = seq.shape[2]
fig, img = plt.subplots(1)

for i in range(cuts-1):
    print(i)
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)
    # if i == 0 or i==99 or i==199 or i==299 or i==399:
        # if i > 30:
    img.imshow(It, cmap='gray')
    # box = patches.Rectangle((rect[0], rect[1]), width, height, linewidth=1, edgecolor='r', facecolor='none')
    img.add_patch(mask)
    plt.show(block=False)
    plt.pause(0.01)
    img.clear()