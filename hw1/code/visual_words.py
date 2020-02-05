import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
from multiprocessing import Pool
from functools import partial
import util


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    shape = img.shape
    H = shape[0]
    W = shape[1]
    filter_responses = np.zeros((H, W, 3*len(filter_scales)*4))
    for i in range(len(filter_scales)):
        for j in range(img.shape[2]):
            filter_responses[:, :, i*12+j:i*12+1+j] = scipy.ndimage.gaussian_filter(img[:,:,j:j+1], filter_scales[i])
            filter_responses[:, :, i*12+3+j:i*12+4+j] = scipy.ndimage.gaussian_laplace(img[:,:,j:j+1], filter_scales[i])
            filter_responses[:, :, i*12+6+j:i*12+7+j] = scipy.ndimage.gaussian_filter1d(img[:,:,j:j+1], filter_scales[i], axis=1, order=1)
            filter_responses[:, :, i*12+9+j:i*12+10+j] = scipy.ndimage.gaussian_filter1d(img[:,:,j:j+1], filter_scales[i], axis=0, order=1)

    # filter_responses = skimage.color.lab2rgb(filter_responses)
    return filter_responses

def compute_dictionary_one_image(img_path, a, opts):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    # read image and convert to LAB colors
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    img = skimage.color.rgb2lab(img)

    # sample random alpha points
    shape = img.shape
    H = shape[0]
    W = shape[1]

    # take care of grayscale image
    if len(shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    hrand = np.random.rand(a)
    wrand = np.random.rand(a)
    hrand = (H*hrand).astype(int)
    wrand = (W*wrand).astype(int)
    img_sample = img[hrand, wrand, :]

    # change img dimensions for extract_filter_responses, and then change back
    img_sample = img_sample.reshape(1, a, 3)
    filter_responses = extract_filter_responses(opts, img_sample)
    return filter_responses[0]

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    # create multiprocessing
    pool = Pool()
    alph = opts.alpha
    for f in range(len(train_files)):
        train_files[f] = join(opts.data_dir, train_files[f])
    compute_partial = partial(compute_dictionary_one_image, a=alph, opts=opts)
    filter_responses = pool.map(compute_partial, train_files)

    # stack the multiprocessed filter_responses together for kmeans
    T = len(filter_responses)
    stacked_filter = np.zeros((alph*T, 12*len(opts.filter_scales)))
    for f in range(T):
        stacked_filter[f*alph:f*alph+alph, :] = filter_responses[f]
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=n_worker).fit(stacked_filter)
    dictionary = kmeans.cluster_centers_
    print(dictionary.shape)

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    shape = img.shape
    wordmap = np.zeros((shape[0], shape[1]))
    filter_responses = extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)
    i = 0
    for row in filter_responses:
        distances = scipy.spatial.distance.cdist(row, dictionary)
        j = 0
        for ys in distances:
            word = np.argmax(ys)
            # print(word)
            wordmap[i, j] = word
            j += 1
        i += 1
    return wordmap

