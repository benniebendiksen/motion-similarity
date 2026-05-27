import torch
import numpy as np


class _SequenceBase:
    """
    Minimal drop-in for keras.utils.Sequence — no Keras/TF dependency required.
    Provides the same interface contract: __len__, __getitem__, on_epoch_end.
    """
    def __len__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError
    def on_epoch_end(self): pass


class SimilarityDataLoader(_SequenceBase):
    def __init__(self, list_similarity_dicts, config, shuffle=False, valid_indices=None):
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

        self.list_tuples_dict_idx_class_tuple = []

        # Track module sizes and start indices for proper batch slicing
        self.module_sizes = []
        self.module_start_indices = []

        all_classes_count = 0

        # First calculate module sizes and start indices
        start_idx = 0
        for i, similarity_dict in enumerate(list_similarity_dicts):
            # Get appropriate valid indices for this animation
            curr_valid_indices = None if valid_indices is None else valid_indices[i]

            # Count how many classes will be included from this module
            module_examples = 0
            for class_tuple in similarity_dict.keys():
                if curr_valid_indices is None or class_tuple in curr_valid_indices:
                    module_examples += 1

            self.module_sizes.append(module_examples)
            self.module_start_indices.append(start_idx)
            start_idx += module_examples

        for i, similarity_dict in enumerate(list_similarity_dicts):
            # Get appropriate valid indices for this animation. A train/val split ensuring mechanism
            curr_valid_indices = None if valid_indices is None else valid_indices[i]

            for _, (class_tuple, value) in enumerate(similarity_dict.items()):
                # Skip neutral for pointing if it somehow exists (defensive programming)
                # if i == 1 and class_tuple == (0, 0, 0, 0):  # Index 1 is pointing
                #     print("Skipping neutral from the construction of dict_similarity_exemplars and list_tuples_dict_idx_class_tuple")
                #     continue

                # Skip if not in valid indices. A train/val split ensuring mechanism
                if curr_valid_indices is not None and class_tuple not in curr_valid_indices:
                    continue

                new_key = (i, class_tuple)
                self.dict_similarity_exemplars[new_key] = value
                self.list_tuples_dict_idx_class_tuple.append(new_key)
                self.class_indexes.append(all_classes_count)
                all_classes_count += 1
        # Each key is a tuple containing the dictionary identifier and the original key (state or drive index id).
        self.num_classes = len(self.class_indexes)
        print(f"SimilarityDataLoader: num classes: {self.num_classes}")

        # batch_size is the number of classes in the dataset subset. There is only one batch per epoch.
        self.batch_size = len(self.dict_similarity_exemplars.keys())
        print(f"SimilarityDataLoader: batch size: {self.batch_size}")
        self._num_batches = 1
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

    # def __getitem__(self, index):
    #     # Collect features for all class tuples and convert to a tensor
    #     batch_features = tf.convert_to_tensor(
    #         [self.dict_similarity_exemplars[class_tuple][0] for class_tuple in self.list_tuples_dict_idx_class_tuple]
    #     )
    #     # Ensure the tensor has a channel dimension (e.g., for compatibility with convolutional models)
    #     batch_features = batch_features[..., tf.newaxis]
    #     # Return the batch of features along with their corresponding class labels
    #     return batch_features, tf.constant(self.class_indexes)

    def __getitem__(self, index):
        if index >= self._num_batches:
            raise StopIteration

        # Convert NumPy arrays directly to PyTorch tensors
        batch_features = torch.from_numpy(
            np.array([self.dict_similarity_exemplars[class_tuple][0] for class_tuple in
                      self.list_tuples_dict_idx_class_tuple])
        ).float()  # Ensure correct dtype for model input

        # Add a channel dimension (for CNN compatibility, e.g., [batch_size, channels, height, width])
        batch_features = batch_features.unsqueeze(-1)  # Equivalent to tf.newaxis

        # Convert class indexes to a PyTorch tensor
        class_labels = torch.tensor(self.class_indexes, dtype=torch.long)

        return batch_features, class_labels

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self._num_batches:
            raise StopIteration

        batch = self.__getitem__(self.current_index)
        self.current_index += 1
        return batch
