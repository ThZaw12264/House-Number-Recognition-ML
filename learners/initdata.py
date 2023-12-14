import numpy as np 
import scipy.io

NUM_TRAIN_IMAGES = 73257
NUM_TEST_IMAGES = 26032
IMAGE_SIZE = 32 * 32 * 3


def initdata():
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