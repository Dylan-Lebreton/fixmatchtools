import numpy as np
from keras.datasets import cifar10


def generate_cifar10_dataset(labeled_data_proportion: float):
    """
    Return (labeled_data_proportion)% of cifar10 train data as labeled data,
    100% of cifar10 train data as unlabeled data and test data from cifar10.
    """

    # retrieve data (additional underscore on test to prevent variables conflicts)
    (x_train, y_train), (x_test_, y_test_) = cifar10.load_data()

    # get total number of examples
    total_examples = x_train.shape[0]

    # compute the number of labeled examples
    labeled_examples = int(np.floor(total_examples * labeled_data_proportion))

    # randomly choose indices for labeled examples
    labeled_indices = np.random.choice(total_examples, size=labeled_examples, replace=False)

    # get labeled dataset
    x_train_lab_res = x_train[labeled_indices]
    y_train_lab_res = y_train[labeled_indices]

    # get unlabeled dataset
    x_train_unlab_res = x_train
    y_train_unlab_res = y_train

    return x_train_lab_res, y_train_lab_res, x_train_unlab_res, y_train_unlab_res, x_test_, y_test_