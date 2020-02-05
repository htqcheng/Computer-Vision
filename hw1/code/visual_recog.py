import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    hist = np.histogram(wordmap, bins=range(K+1), density=True)
    # unique, counts = np.unique(wordmap, return_counts=True)
    # print(dict(zip(unique, counts)))
    print(sum(hist[0]))
    # # print(wordmap)
    # print(K, len(hist[0]))
    return hist[0]

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    # calculate histogram of the last layer
    cell_len = 2**L
    H, W = wordmap.shape
    h = H//cell_len
    w = W//cell_len
    cell_hists = np.zeros((cell_len, cell_len, K))
    for i in range(cell_len):
        for j in range(cell_len):
            cell = wordmap[i*h:i*h+h, j*w:j*w+w]
            cell_hist = get_feature_from_wordmap(opts, cell)
            cell_hists[i, j, :] = cell_hist

    # store it
    hist_all = np.zeros(int(K*(4**(L+1)-1)/3))
    weight = 2**(-1)
    hist_all[-(K*cell_len**2):] = cell_hists.reshape(-1)*weight
    print(sum(hist_all))

    # sum up last layer's histograms for the next layer's histogram and store it
    for l in range(L-1, -1, -1):
        cell_hists2 = np.zeros((2**l, 2**l, K))
        cell_len = 2**l
        for i in range(cell_len):
            for j in range(cell_len):
                first_cell = cell_hists[2*i, 2*j]
                second_cell = cell_hists[2*i, 2*j+1]
                third_cell = cell_hists[2*i+1, 2*j]
                fourth_cell = cell_hists[2*i+1, 2*j+1]
                cell_hists2[i, j, :] = (first_cell+second_cell+third_cell+fourth_cell)/4
        # calculate weight
        if l == 0:
            weight = 2**(-L)
            insert_location = 0
        else:
            weight = 2**(l-L-1)
            insert_location = 0
            for m in range(l):
                insert_location += 4 ** m
        # get the location to insert into hist_all
        insert_location *= K
        hist_all[insert_location:insert_location+len(cell_hists2.reshape(-1))] = cell_hists2.reshape(-1) * weight
        # set cell_hists to be the next layer
        cell_hists = cell_hists2
    print(sum(hist_all))
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    pass

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    pass

    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    N, K = histograms.shape
    sim = np.zeros(N)
    for i in range(len(histograms)):
        sim[i] = np.sum(np.minimum(histograms[i], word_hist))
    return sim
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    pass

