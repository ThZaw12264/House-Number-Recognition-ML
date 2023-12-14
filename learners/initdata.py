import numpy as np 
import scipy.io

NUM_TRAIN_IMAGES = 73257
NUM_TEST_IMAGES = 26032
NUM_CLASSES = 10
IMAGE_SIZE = 32 * 32 * 3

def cnndata():
    # im_trx = np.zeros(shape=(NUM_TRAIN_IMAGES,32,32,3))
    # im_try = np.zeros(shape=(NUM_TRAIN_IMAGES,1))

    # im_testx = np.zeros(shape=(NUM_TEST_IMAGES,32,32,3))
    # im_testy = np.zeros(shape=(NUM_TEST_IMAGES,1))

    train_mat =  scipy.io.loadmat("../data/train_32x32.mat")
    test_mat =  scipy.io.loadmat("../data/test_32x32.mat")


    im_trx = train_mat['X']
    im_try = train_mat['y']
    im_testx = test_mat['X']
    im_testy = test_mat['y']

    im_trx = np.moveaxis(im_trx, -1, 0) 
    im_testx = np.moveaxis(im_testx, -1, 0)

    im_trx = im_trx.astype('float32') / 255.0
    im_testx = im_testx.astype('float32') / 255.0
    im_try = im_try - 1
    im_testy = im_testy - 1

    return im_trx, im_testx, im_try, im_testy

def lrdata():
    im_trx = np.zeros(shape=(NUM_TRAIN_IMAGES,IMAGE_SIZE))
    im_try = np.zeros(shape=(NUM_TRAIN_IMAGES,1))

    im_testx = np.zeros(shape=(NUM_TEST_IMAGES,IMAGE_SIZE))
    im_testy = np.zeros(shape=(NUM_TEST_IMAGES,1))

    train_mat =  scipy.io.loadmat("../data/train_32x32.mat")
    test_mat =  scipy.io.loadmat("../data/test_32x32.mat")


    X_train = train_mat['X'].transpose(3, 0, 1, 2) 
    X_test = test_mat['X'].transpose(3, 0, 1, 2)

    im_trx = X_train.reshape(NUM_TRAIN_IMAGES, -1)
    im_testx = X_test.reshape(NUM_TEST_IMAGES, -1)

    im_try = train_mat['y']
    im_testy = test_mat['y']

    return im_trx, im_testx, im_try, im_testy