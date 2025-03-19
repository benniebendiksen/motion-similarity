import keras
import concurrent
import os.path
import numpy as np
from glob import glob
import conf_2


def unison_shuffling(batch_features, batch_labels):
    p = np.random.permutation(len(batch_features))
    return batch_features[p], batch_labels[p]


class MotionDataGenerator(keras.utils.Sequence):
    def __init__(self, list_batch_ids, labels, batch_size=conf.batch_size_efforts_network, exemplar_dim=(100, 87),
                 exemplars_dir=conf.effort_network_exemplars_dir, shuffle=True):
        self.exemplar_dim = exemplar_dim
        self.batch_size = batch_size
        self.batch_group_size = 16
        self.grouped_batches = None
        self.max_grouped_batches_idx = None
        self.labels = labels
        self.list_batch_ids = list_batch_ids
        self.num_batches = len(self.list_batch_ids)
        self.exemplars_dir = exemplars_dir
        self.shuffle = shuffle

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.list_batch_ids)

    def __len__(self):
        #num_batches
        return len(self.list_batch_ids)

    def generator(self, index):

        def _load_batch(i, idx):
            # single batch fetching w
            path = glob(os.path.join(self.exemplars_dir, f'*_{idx}.npy'))
            if len(path) != 1:
                assert False, f"Error for id {idx}, found path for batch must be unique — {path}!"
            batch_features = np.load(path[0])
            assert self.labels[idx] is not None, f"missing label at idx: {idx}"
            # return batch_features, np.array(self.labels[idx])
            return batch_features, self.labels[idx]

        def read_async_batch_files(subset_batch_ids):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_load_batch, i, idx) for i, idx in enumerate(subset_batch_ids)]
                return [fut.result() for fut in futures]

        modulo = index % self.batch_group_size
        if modulo == 0:
            for batch_group_idx in range(0, self.num_batches, self.batch_group_size):
                # truncate group upper limit if it exceeds num_batches
                group_upper_lim = min(batch_group_idx+self.batch_group_size, self.num_batches)
                self.grouped_batches = read_async_batch_files(self.list_batch_ids[batch_group_idx:group_upper_lim])
                # correct for end fetching corner case where fetched grouped_batches is less than batch_group_size
                self.max_grouped_batches_idx = len(self.grouped_batches) - 1
        grouped_batches_idx = min(modulo, self.max_grouped_batches_idx)
        return self.grouped_batches[grouped_batches_idx][0], self.grouped_batches[grouped_batches_idx][1]

    def __getitem__(self, index):
        # (64, 100, 87), (64, 4)
        batch_features, batch_labels = self.generator(index)
        batch_features, batch_labels = unison_shuffling(batch_features, batch_labels)
        return batch_features, batch_labels

