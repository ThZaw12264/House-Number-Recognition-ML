import numpy as np 
import matplotlib.pyplot as plt
import initdata as id

from sklearn.metrics import zero_one_loss

from sklearn.linear_model import LogisticRegression

im_trx, im_testx, im_try, im_testy = id.initdata()

seed = 100101

im_trx_subset = im_trx[:10000]
im_try_subset = im_try[:10000]

# Fit training data to learner and predict
learner = LogisticRegression(random_state=seed,C=0.000001,max_iter=500)
learner.fit(im_trx_subset, im_try_subset.ravel())
pred_train = learner.predict(im_trx_subset)
pred_test = learner.predict(im_testx)
print(f'Training Error Rate: {zero_one_loss(pred_train, im_try_subset)}')
print(f'Test Error Rate: {zero_one_loss(pred_test, im_testy)}')


def plot_regularization():
    C_vals = [0.000000001,0.00000001,0.0000001,.000001,.00001,.0001]; 
    tr_er = []
    te_er = []

    for c in C_vals:
        learner = LogisticRegression(random_state=seed, max_iter=1000, C=c)
        learner.fit(im_trx_subset, im_try_subset.ravel())

        # Compute the training and test error rates
        tr_er.append(zero_one_loss(learner.predict(im_trx_subset), im_try_subset.ravel()))
        te_er.append(zero_one_loss(learner.predict(im_testx), im_testy.ravel()))
        print("Trained")

    # Plot the resulting performance as a function of C
    print(tr_er)
    print(te_er)
    plt.semilogx(C_vals, tr_er, 'r', C_vals, te_er, 'g')
    plt.xlabel('Regularization (C)')
    plt.ylabel('Error Rate')
    plt.legend(['Train Error','Test Error'])
    plt.show()
    # 0.000001 is best C with 0.7729 test error rate and 0.668 training error rate


def plot_max_iter():
    max_iters = [100,500,1000,2000,4000,8000]
    tr_er = []
    te_er = []

    for max_iter in max_iters:
        learner = LogisticRegression(random_state=seed, max_iter=max_iter, C=0.000001)
        learner.fit(im_trx_subset, im_try_subset.ravel())

        # Compute the training and test error rates
        tr_er.append(zero_one_loss(learner.predict(im_trx_subset), im_try_subset.ravel()))
        te_er.append(zero_one_loss(learner.predict(im_testx), im_testy.ravel()))
        print("Trained")

    # Plot the resulting performance as a function of max_iter
    print(tr_er)
    print(te_er)
    plt.semilogx(max_iters, tr_er, 'r', max_iters, te_er, 'g')
    plt.xlabel('Max iterations')
    plt.ylabel('Error Rate')
    plt.legend(['Train Error','Test Error'])
    plt.show()
    # 500 is best max_iter: lowest test error rate while preserving speed


def plot_train_sizes():
    train_sizes = [500,1000,5000,10000,25000,50000,id.NUM_TRAIN_IMAGES]
    tr_er = []
    te_er = []

    for train_size in train_sizes:
        learner = LogisticRegression(random_state=seed, max_iter=500, C=0.000001)
        learner.fit(im_trx[:train_size], im_try[:train_size].ravel())

        # Compute the training and test error rates
        tr_er.append(zero_one_loss(learner.predict(im_trx[:train_size]), im_try[:train_size].ravel()))
        te_er.append(zero_one_loss(learner.predict(im_testx), im_testy.ravel()))
        print("Trained")

    # # Plot the resulting performance as a function of training size
    print(tr_er)
    print(te_er)
    plt.semilogx(train_sizes, tr_er, 'r', train_sizes, te_er, 'g')
    plt.xlabel('Training Size')
    plt.ylabel('Error Rate')
    plt.legend(['Train Error','Test Error'])
    plt.show()
    # Test error rate of 0.7530347264904733 and training error rate of 0.6957969886836752 with training all data


def plot_coefficients():
    # Heatmap of coefficients for each class
    fig, ax = plt.subplots(1,10, figsize=(18,8))
    mu = learner.coef_.mean(0).reshape(32,32,3)
    for i in range(10):
        coef_norm = (learner.coef_[i, :].reshape(32, 32, 3) - mu)
        coef_norm -= np.min(coef_norm)
        coef_norm /= np.max(coef_norm)

        ax[i].imshow(coef_norm,cmap='seismic',vmin=0,vmax=255)
        ax[i].set_title(f'Class {(i+1)%10}')
        ax[i].axis('off')
    plt.show()

plot_train_sizes()
plot_coefficients()


# def get_wrong_predictions():
#     # Get wrong predictions and produce their images
#     wrong = []
#     for i in range(pred_test.size):
#          if pred_test[i] != y_test[i]:
#               wrong.append(i)

#     fig, ax = plt.subplots(10,4, figsize=(32,2))
#     fig.subplots_adjust(hspace=1)
#     for i, ind in enumerate(wrong):
#          j = i // 4
#          k = i % 4
#          ax[j,k].imshow(im_test[ind].reshape(32,32), cmap ="gray", vmin=0, vmax=255)
#          ax[j,k].set_title(f'Predicted {int(pred_test[ind])}', size=12)
#          ax[j,k].axis('off')
#     plt.show()