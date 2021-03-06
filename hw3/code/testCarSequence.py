import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../../../CV_Large_data/hw3/data/carseq.npy")
rect = [59, 116, 145, 151]
cuts = seq.shape[2]
carseqrects = []
carseqrects.append(rect)
# fig, img = plt.subplots(1)

for i in range(cuts-1):
    print(i)
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    if i == 0 or i==99 or i==199 or i==299 or i==399:
        # if i > 30:
        fig, img = plt.subplots(1)
        img.imshow(It, cmap='gray')
        width = rect[2]-rect[0]
        height = rect[3]-rect[1]
        box = patches.Rectangle((rect[0], rect[1]), width, height, linewidth=1, edgecolor='r', facecolor='none')
        img.add_patch(box)
        plt.axis('off')
        plt.show(block=False)
        plt.savefig('car_' + str(i+1), bbox_inches='tight')
        plt.pause(0.1)
        img.clear()

    rect[0] += p[0]
    rect[2] += p[0]
    rect[1] += p[1]
    rect[3] += p[1]
    carseqrects.append(rect)
    # plt.show()
carseqrects = np.asarray(carseqrects)
print(carseqrects.shape)
np.save("carseqrects", carseqrects)