import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
import matplotlib.pyplot as plt
from opts import get_opts
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
from multiprocessing import Pool
from functools import partial

def pool_helper(cv_cover, opts, new_source_w, frames):
    book_frame = frames[0]
    source_frame = frames[1]
    h, w, _ = cv_cover.shape
    matches, locs1, locs2 = matchPics(book_frame, cv_cover, opts)
    print(len(matches))
    if len(matches) < 4:
        return book_frame
    locs1_matched = locs1[matches[:, 0]]
    locs2_matched = locs2[matches[:, 1]]
    H, _ = computeH_ransac(locs1_matched, locs2_matched, opts)
    # crop the source frame to be the same size as cv_cover
    source_frame = cv2.resize(source_frame, (new_source_w, int(h*(4/3))))
    cropped_source_frame = source_frame[int(h*(2/3)-h//2):int(h*(2/3)+h//2), (new_source_w // 2 - w // 2):(new_source_w // 2 + w // 2)]
    composite_frame = compositeH(H, book_frame, cropped_source_frame)
    return composite_frame


def main():
    #Write script for Q3.1
    opts = get_opts()
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    h, w, _ = cv_cover.shape
    print('read cover')
    book_frames = loadVid("../data/book.mov")
    book_h, book_w, _ = book_frames[0].shape
    print('loaded book')
    source_frames = loadVid("../data/ar_source.mov")
    print('loaded source')
    source_h, source_w, _ = source_frames[0].shape
    new_source_w = int(source_w/source_h*h*(4/3))
    fps = len(book_frames)/21


    # for multiprocessing pool implementation
    frames = []
    for f in range(len(book_frames)):
        if f < len(source_frames):
            frame = [book_frames[f], source_frames[f]]
        else:
            frame = [book_frames[f], source_frames[-1]]
        frames.append(frame)
    pool = Pool()
    compute_partial = partial(pool_helper, cv_cover, opts, new_source_w)
    out_frames = pool.map(compute_partial, frames)
    out = cv2.VideoWriter('../result/ar2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (book_w, book_h))
    for i in range(len(out_frames)):
        out.write(out_frames[i])
        print('writing loop' + str(i))


if __name__ == '__main__':
    main()

# # for regular outfile writing
# out = cv2.VideoWriter('../result/ar.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (book_w, book_h))
# # loop through every frame in book.mov
# for i in range(len(book_frames)):
#     print('start loop')
#     # find the homography matrix required for that frame
#     matches, locs1, locs2 = matchPics(book_frames[i], cv_cover, opts)
#     if len(matches) < 4:
#         continue
#     N = matches.shape[0]
#     locs1_matched = locs1[matches[:, 0]]
#     locs2_matched = locs2[matches[:, 1]]
#     H, _ = computeH_ransac(locs1_matched, locs2_matched, opts)
#     # crop the source frame to be the same size as cv_cover
#     if i < len(source_frames):
#         source_frame = source_frames[i]
#     else:
#         source_frame = source_frames[-1]
#     source_frame = cv2.resize(source_frame, (new_source_w, h*(4/3)))
#     cropped_source_frame = source_frame[int(h*(2/3)-h//2):int(h*(2/3)+h//2), (new_source_w//2 - w//2):(new_source_w//2 + w//2)]
#     composite_frame = compositeH(H, book_frames[i], cropped_source_frame)
#     out.write(composite_frame)
#     print(i)
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
