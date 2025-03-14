import keras
import tensorflow as tf
import numpy as np


class SimilarityDataLoader(keras.utils.Sequence):
    def __init__(self, list_similarity_dicts, config, shuffle=False):
        """
        self.dict_similarity_exemplars, the data structure that represents our dataset, gets
        structured as: k,v = (dict_idx, class_tuple), tensor (shape: 137, 88) where all tensors are of same shape
        self.class_indexes serve as the single batch "labels" and are comprised of integers 0 to 56 repeated three times over
        * Mapping between class_tuple and class_index occurs here.
        self.list_tuples_dict_idx_class_tuple: list of elements: (dict_idx, class_tuple). 3 * 57 total elements
        """
        super().__init__()
        print("initializing similarity data loader")
        self.config = config
        self.exemplar_dim = self.config.similarity_exemplar_dim
        self.shuffle = shuffle
        self.dict_similarity_exemplars = {}
        self.class_indexes = []
        # self.class_indexes, _unused_var = zip(
        #     *[(index + 1, value) for index, value in enumerate(list(list_similarity_dicts[0].keys()))])
        self.list_tuples_dict_idx_class_tuple = []
        all_classes_count = 0
        for i, similarity_dict in enumerate(list_similarity_dicts):
            for _, (class_tuple, value) in enumerate(similarity_dict.items()):
                new_key = (i, class_tuple)
                self.dict_similarity_exemplars[new_key] = value
                self.list_tuples_dict_idx_class_tuple.append(new_key)
                self.class_indexes.append(all_classes_count)
                all_classes_count += 1
        # Each key is a tuple containing the dictionary identifier and the original key (state or drive index id).
        self.num_classes = len(self.class_indexes)
        print(f"SimilarityDataLoader: num classes: {self.num_classes}")

        # self._num_batches = len(self.list_class_tuples) // self.batch_size
        # if len(self.list_class_tuples) % self.batch_size != 0:
        #     self._num_batches += 1
        self.batch_size = len(self.dict_similarity_exemplars.keys())
        print(f"SimilarityDataLoader: batch size: {self.batch_size}")
        self._num_batches = 1
        # self._num_batches = len(self.dict_similarity_exemplars[next(iter(self.dict_similarity_exemplars.keys()))][0])
        print(f"SimilarityDataLoader: num batches: {self._num_batches}")
        # self.exemplar_idx = random.randint(0, self.num_batches-1)
        self.exemplar_idx = 0

    def unison_shuffling(self):
        # generate random batch num index, to be applied to all dict class lists, and unison shuffle
        # class and class_idx order (a within batch shuffling)
        # self.exemplar_idx = random.randint(1, self._num_batches - 1)
        # p = list(np.random.permutation(self.num_classes))
        # # the tight coupling between class_indexes and list_class_tuples allows for class index (batch label) to
        # # relevant class exemplar (batch feature) mapping
        # self.class_indexes = [self.class_indexes[i] for i in p]
        # self.list_tuples_dict_idx_class_tuple = [self.list_tuples_dict_idx_class_tuple[i] for i in p]

        p = np.random.permutation(len(self.class_indexes))
        self.class_indexes = [self.class_indexes[i] for i in p]
        self.list_tuples_dict_idx_class_tuple = [self.list_tuples_dict_idx_class_tuple[i] for i in p]

    def on_epoch_end(self):
        if self.shuffle:
            self.unison_shuffling()
        self.exemplar_idx = 0

    def __len__(self):
        # num_batches
        return self._num_batches

    # def __getitem__(self, index):
    #     batch_features = tf.stack([self.dict_similarity_exemplars[class_tuple][0] for class_tuple in
    #                                self.list_tuples_dict_idx_class_tuple])
    #     if len(batch_features.shape) == 3:
    #         # Add channel dimension if it's missing
    #         print("extending batch shape for channel dimension")
    #         batch_features = tf.expand_dims(batch_features, -1)
    #     batch_labels = tf.constant(self.class_indexes)
    #     return batch_features, batch_labels

    def __getitem__(self, index):
        # Collect features for all class tuples and convert to a tensor
        batch_features = tf.convert_to_tensor(
            [self.dict_similarity_exemplars[class_tuple][0] for class_tuple in self.list_tuples_dict_idx_class_tuple]
        )
        # Ensure the tensor has a channel dimension (e.g., for compatibility with convolutional models)
        batch_features = batch_features[..., tf.newaxis]
        # Return the batch of features along with their corresponding class labels
        return batch_features, tf.constant(self.class_indexes)
