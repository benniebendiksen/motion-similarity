import conf_2
from collections import Counter
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.api.callbacks import CSVLogger
import datetime
import numpy as np


class Utilities:
    """
    An utilities class defined by attributes and methods amenable to tensorboard networks
    Inherited by network classes
    """
    def __init__(self):
        # if callback used, run tensorboard --logdir path_to_current_dir/logs/fit to view files created during training
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.csv_logger = CSVLogger('log.csv', append=True, separator=';')
        self.train_split = 0.8
        self.train_ds = None
        self.test_ds = None
        self.classes = None

    def partition_dataset(self, x, y, train_split=0.9, seed=11):
        print(f"dataset shape: {x.shape}")
        train_size = int(train_split * x.shape[0])
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        train_ds = ds.take(train_size)
        test_ds = ds.skip(train_size)
        train_ds = train_ds.shuffle(60000, reshuffle_each_iteration=True)
        test_ds = test_ds.shuffle(60000, reshuffle_each_iteration=True)
        print(type(train_ds))
        return train_ds, test_ds

    def make_classes_from_labels(self, labels):
        """
        Convert array of labels to array of classes denoted by number
        :param labels: np.array: ground truths for network
        :return: np.array
        """
        labels_classes = np.empty((labels.shape[0], 1))
        print(f"y shape: {labels_classes.shape}")
        # labels_map={label: class_num}
        labels_map = {label: class_num for (class_num, label) in enumerate(set(tuple(x) for x in labels.tolist()))}
        # get class_num to populate new labels/classes array
        for idx, label in enumerate(labels.tolist()):
            labels_classes[idx] = labels_map[tuple(label)]
        self.classes = labels_classes

    def train_test_split(self, x, y, train_split=0.8, seed=0):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.shuffle(buffer_size=conf.buffer_size, seed=seed)
        train_size = int(train_split * x.shape[0])
        train_ds = ds.take(train_size)
        test_ds = ds.skip(train_size)
        y_train_list = [elem[1] for elem in train_ds.as_numpy_iterator()]
        y_test_list = [elem[1] for elem in test_ds.as_numpy_iterator()]
        print(f"train classes #: {len(np.unique(y_train_list))}")
        print(f"test classes #: {len(np.unique(y_test_list))}")
        self.train_ds = train_ds.shuffle(60000).batch(conf.batch_size_efforts_network).repeat()
        self.test_ds = test_ds.shuffle(60000).batch(conf.batch_size_efforts_network).repeat()

    @staticmethod
    def visualize_class_distribution(train_ds, test_ds, train_title, test_title):
        """
        Generate label frequencies barplots for training and testing sets
        :param train_ds: np.array or tensorflow dataset: training ground truths for network or entire training ds
        :param test_ds: np.array or tensorflow dataset: test ground truths for network or entire testing ds
        :param train_title: str: title for training barplot
        :param test_title: str: title for testing barplot
        :return: None
        """
        train_test_labels = []
        if type(train_ds).__module__ == np.__name__:
            if train_ds.ndim == 2 or test_ds.ndim == 2:
                train_ds = train_ds.flatten()
                test_ds = test_ds.flatten()
            train_test_labels.append(train_ds)
            train_test_labels.append(test_ds)
        else:
            train_test_labels.append(np.concatenate([y for x, y in train_ds], axis=0))
            train_test_labels.append(np.concatenate([y for x, y in test_ds], axis=0))
        for count, labels_array in enumerate(train_test_labels):
            labels_array.sort()
            labels, counts = zip(*Counter(labels_array).items())
            indexes = np.arange(len(labels))
            width = 1
            plt.bar(indexes, counts, width)
            plt.xticks(indexes + width * 0.5, labels)
            plt.xlabel('class index')
            plt.ylabel('frequency')
            if count == 0:
                plt.title(f"{train_title} with n = {len(labels_array)}")
            else:
                plt.title(f"{test_title} with n = {len(labels_array)}")
            plt.show(block=False)
            plt.pause(3)
            plt.close()
