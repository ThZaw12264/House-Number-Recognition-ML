import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.io

from sklearn.metrics import zero_one_loss

from sklearn.linear_model import LogisticRegression

NUM_TRAIN_IMAGES = 73257
NUM_TEST_IMAGES = 26032
IMAGE_SIZE = 32 * 32 * 3


ima_trx = np.zeros(shape=(NUM_TRAIN_IMAGES,IMAGE_SIZE))
ima_try = np.zeros(shape=(NUM_TRAIN_IMAGES,1))

ima_testx = np.zeros(shape=(NUM_TEST_IMAGES,IMAGE_SIZE))
ima_testy = np.zeros(shape=(NUM_TEST_IMAGES,1))

train_mat =  scipy.io.loadmat("../data/train_32x32.mat")
test_mat =  scipy.io.loadmat("../data/test_32x32.mat")


X_train = train_mat['X'].transpose(3, 0, 1, 2) 
X_test = test_mat['X'].transpose(3, 0, 1, 2)

ima_trx = X_train.reshape(NUM_TRAIN_IMAGES, -1)
ima_testx = X_test.reshape(NUM_TEST_IMAGES, -1)

ima_try = train_mat['y']
ima_testy = test_mat['y']

print(ima_trx[0])
plt.imshow(ima_trx[0].reshape(32,32,3))
plt.show()


