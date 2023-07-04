import numpy as np
from keras.datasets import cifar10
from matplotlib import pyplot as plt

LABELS_MAPPING = {0: "avion", 1: "automobile", 2: "oiseau", 3: "chat", 4: "cerf",
                  5: "chien", 6: "crapaud", 7: "cheval", 8: "bateau", 9: "camion"}

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

def to_labels(y_pred):
    """
    Convertit les probabilités prédites en classes.
    """
    # obtenir l'index de la classe prédite (la plus probable)
    class_idx = np.argmax(y_pred, axis=-1)
    # reshape
    class_idx = class_idx.reshape(-1, 1)
    return class_idx.astype(int)


def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, suptitle=""):

    fig, axs = plt.subplots(2, 1, sharex='col', sharey='row')

    # Loss
    axs[0].plot(train_losses, label='Loss on training set')
    axs[0].plot(test_losses, label='Loss on test set')
    min_val_loss_epoch = np.argmin(test_losses)
    min_val_loss = min(test_losses)
    axs[0].set_title(f'Evolution of train and test losses over epochs (best test loss = {round(min_val_loss,2)} at epoch {min_val_loss_epoch})')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Accuracy
    axs[1].plot(train_accuracies, label='Accuracy on training set')
    axs[1].plot(test_accuracies, label='Accuracy on test set')
    max_val_acc_epoch = np.argmax(test_accuracies)
    max_val_acc = max(test_accuracies)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title(f'Evolution of train and test accuracies over epochs (best test accuracy = {round(max_val_acc, 2)} at epoch {max_val_acc_epoch})')
    axs[1].legend()

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_train_true, y_train_pred, y_test_true, y_test_pred):
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

  fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  # train data
  cm_train = confusion_matrix(y_train_true, to_labels(y_train_pred))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=LABELS_MAPPING.values())
  disp.plot(cmap='viridis', ax=axs[0], colorbar=False)
  axs[0].set_title(f'Confusion matrix on train data')
  plt.setp(disp.ax_.xaxis.get_majorticklabels(), rotation=90)

  # test data
  cm_test = confusion_matrix(y_test_true, to_labels(y_test_pred))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=LABELS_MAPPING.values())
  disp.plot(cmap='viridis', ax=axs[1], colorbar=False)
  axs[1].set_title('Confusion matrix on test data')
  plt.setp(disp.ax_.xaxis.get_majorticklabels(), rotation=90)

  plt.tight_layout()
  plt.show()
