import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

# seq = np.load("../../../CV_Large_data/hw3/data/carseq.npy")
seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

rect_iter = rect.copy()
cuts = seq.shape[2]
carseqrects = [rect]
T1 = seq[:, :, 0]
# fig, img = plt.subplots(1)
p_star = np.zeros(2)

for i in range(cuts-1):
    print(i)
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, rect_iter, threshold, num_iters)
    p_iter = p_star + p
    if i == 0 or i==99 or i==199 or i==299 or i==399:
    # if i > 30:
        fig, img = plt.subplots(1)
        img.imshow(It, cmap='gray')
        width = rect_iter[2]-rect_iter[0]
        height = rect_iter[3]-rect_iter[1]
        box = patches.Rectangle((rect_iter[0], rect_iter[1]), width, height, linewidth=1, edgecolor='b', facecolor='none')
        img.add_patch(box)
        plt.axis('off')
        plt.show(block=False)
        plt.savefig('car_template_' + str(i + 1), bbox_inches='tight')
        plt.pause(0.1)
        img.clear()
    
    # rect[0] += p[0]
    # rect[2] += p[0]
    # rect[1] += p[1]
    # rect[3] += p[1]
    p_star = LucasKanade(T1, It1, rect, threshold, num_iters, p_iter)
    if (np.linalg.norm(p_star - p_iter) <= template_threshold):
        rect_iter[0] = rect[0] + p_star[0]
        rect_iter[2] = rect[2] + p_star[0]
        rect_iter[1] = rect[1] + p_star[1]
        rect_iter[3] = rect[3] + p_star[1]
    else:
        rect_iter[0] = rect[0] + p_iter[0]
        rect_iter[2] = rect[2] + p_iter[0]
        rect_iter[1] = rect[1] + p_iter[1]
        rect_iter[3] = rect[3] + p_iter[1]
    carseqrects.append(rect_iter)
    # plt.show()
carseqrects = np.asarray(carseqrects)
print(carseqrects.shape)
np.save("carseqrects-wcrt", carseqrects)