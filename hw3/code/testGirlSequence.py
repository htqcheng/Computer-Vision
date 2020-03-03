import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

# seq = np.load("../../../CV_Large_data/hw3/data/girlseq.npy")
seq = np.load("/home/htqcheng/CV_Large_data/hw3/data/girlseq.npy")
rect = [280, 152, 330, 318]

cuts = seq.shape[2]
girlseqrects = []
girlseqrects.append(rect)
fig, img = plt.subplots(1)

for i in range(cuts-1):
    print(i)
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    # if i == 0 or i==99 or i==199 or i==299 or i==399:
        # if i > 30:
    img.imshow(It, cmap='gray')
    width = rect[2]-rect[0]
    height = rect[3]-rect[1]
    box = patches.Rectangle((rect[0], rect[1]), width, height, linewidth=1, edgecolor='r', facecolor='none')
    img.add_patch(box)
    plt.show(block=False)
    plt.pause(0.01)
    img.clear()

    rect[0] += p[0]
    rect[2] += p[0]
    rect[1] += p[1]
    rect[3] += p[1]
    girlseqrects.append(rect)
    # plt.show()
girlseqrects = np.asarray(girlseqrects)
print(girlseqrects.shape)
np.save("girlseqrects", girlseqrects)
