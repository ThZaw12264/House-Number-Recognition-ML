import numpy as np 
import matplotlib.pyplot as plt
import initdata as id

from sklearn.metrics import zero_one_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

im_trx, im_testx, im_try, im_testy = id.initdata()