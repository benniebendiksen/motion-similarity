import numpy as np
import tensorflow as tf
import pickle
import math


def print_dict_structure(d, indent=0):
    if d is None:
        print("dict_similarity_exemplars is None, returning")
        return
    # print(f"printing structure of dict with key: {d.keys()[0]}")
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key} (dict):")
            print_dict_structure(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {type(value).__name__}")

class Batches:
    """
    A singleton class that collects all batching logic for both the effort and similarity networks.

    Attributes:
        batch_size (int): The batch size for the efforts network.
        efforts_network_exemplar_dim (tuple): The dimension of an exemplar for the efforts network.
        sample_idx (int): The current effort network exemplar index.
        batch_idx (int): The current effort network batch index.
        state_drive_exemplar_idx: The index for a given state-class (for similarity network).
        current_batch_exemplar (dict): A dictionary to store the current effort network batch.
        list_batch_efforts (list): A list to collect effort network batches.
        dict_efforts_labels (dict): A dictionary to store efforts labels.
        dict_similarity_exemplars (dict): A dictionary to store similarity class exemplars.
        sliding_window_start_index (int): The start index for the sliding window when convolved across a given motion
        file.
    """
    _self = None

    def __new__(cls, config):
        if cls._self is None:
            # print("Initializing Batches class")
            cls._self = super(Batches, cls).__new__(cls)
        return cls._self

    def __init__(self, config):
        self.config = config
        self.batch_size = self.config.batch_size_efforts_network
        self.efforts_network_exemplar_dim = self.config.effort_network_exemplars_dir
        self.sample_idx = 0
        self.batch_idx = 0
        self.state_drive_exemplar_idx = None
        self.current_batch_exemplar = {0: []}
        self.list_batch_efforts = []
        self.dict_efforts_labels = {0: np.zeros((self.batch_size, 4))}
        self.dict_similarity_exemplars = self._generate_similarity_classes_exemplars_dict()
        self.sliding_window_start_index = self.config.time_series_size

    def append_efforts_batch_and_labels(self, exemplar):
        """
        invoked in organize_synthetic_data:prep_all_data_for_training:apply_moving_window() as well as extend_final_batch()
        Append an exemplar to the current batch and its efforts to the list of batch efforts.

        Args:
            exemplar (numpy.ndarray): The exemplar to append.

        Returns:
            None
        """
        # store efforts label
        self.list_batch_efforts.append(exemplar[0][0:4])
        # drop efforts + anim name columns
        exemplar = np.delete(exemplar, slice(5), axis=1)
        assert exemplar.shape == self.efforts_network_exemplar_dim, (f"Exemplar shape: {exemplar.shape} differs from expected:"
                                                             f" {self.efforts_network_exemplar_dim}")
        self.current_batch_exemplar[self.batch_idx].append(exemplar)
        self.sample_idx += 1

    def store_efforts_batch(self):
        """
        Store the current batch and reset batch-related data.

        Returns:
            None
        """
        self.dict_efforts_labels[self.batch_idx] = np.array(self.list_batch_efforts)
        assert len(self.dict_efforts_labels[self.batch_idx]) == 64, f"Len of efforts dict at batch idx " \
                                                                    f"{self.batch_idx} is {len(self.dict_efforts_labels[self.batch_idx])}"

        batch_array = np.array(self.current_batch_exemplar[self.batch_idx])
        np.save(self.config.effort_network_exemplars_dir + 'batch_' + str(self.batch_idx) + '.npy',
                batch_array)
        print(
            f"Stored batch num {self.batch_idx}. Size: {batch_array.shape}.  exemplar count:"
            f" {self.sample_idx}")
        self.tmp_exemplar = batch_array[0]
        self.batch_idx += 1
        self.current_batch_exemplar[self.batch_idx] = []
        self.list_batch_efforts = []
        self.dict_efforts_labels[self.batch_idx] = np.zeros((self.batch_size, 4))

    def append_to_end_file_exemplar(self, exemplar, make_whole_exemplar=False):
        """
        Ensure last exemplar in file is of length conf.time_series_size.

        Args:
            exemplar (numpy.ndarray): The exemplar to append.
            make_whole_exemplar (bool): If True, make the exemplar a whole exemplar.

        Returns:
            numpy.ndarray: The concatenated matrix.
        """
        target_row_num = self.config.time_series_size
        last_row = exemplar[-1]
        if make_whole_exemplar:
            repeats = target_row_num
        else:
            repeats = target_row_num - exemplar.shape[0]
        extended_rows = np.tile(last_row, (repeats, 1))
        concatenated_matrix = np.concatenate((exemplar, extended_rows), axis=0)
        # print(f"make_whole_exemplar: {make_whole_exemplar}, concatenated matrix shape:"
        #       f"{concatenated_matrix.shape}")
        return concatenated_matrix

    def extend_final_batch(self, exemplar):
        """
        Extend the last batch with exemplars to match batch size.

        Args:
            exemplar (numpy.ndarray): The final exemplar.

        Returns:
            None
        """
        # exemplar = np.delete(end_directory_exemplar, slice(5), axis=1)
        last_row = exemplar[-1]
        repeats = self.config.time_series_size
        # print(f"extend_final_batch: size: {self.current_batch_exemplar[self.batch_idx]}, "
        #       f"last_row shape: {last_row.shape}, repeats: {repeats}")
        while len(self.current_batch_exemplar[self.batch_idx]) < self.config.batch_size_efforts_network:
            new_exemplar = np.tile(last_row, (repeats, 1))
            self.sample_idx += 1
            self.append_efforts_batch_and_labels(new_exemplar)
        print(f"2 extend_final_batch: final batch size: {len(self.current_batch_exemplar[self.batch_idx])}")
        self.store_efforts_batch()

    def store_effort_labels_dict(self):
        """
         Write out the dictionary of effort labels.

         Returns:
             None
         """
        try:
            if len(self.dict_efforts_labels) > 0:
                self.dict_efforts_labels.pop(self.batch_idx)
                self.current_batch_exemplar.pop(self.batch_idx)
                self.batch_idx -= 1
            print(f"storing dict_efforts_labels with len {len(self.dict_efforts_labels)} with dict of batches size"
                  f" {len(self.current_batch_exemplar)}")
            with open(self.config.effort_network_exemplars_dir + '/labels_dict.pickle', 'wb') as handle:
                pickle.dump(self.dict_efforts_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"storing {len(self.dict_efforts_labels.keys())} batch labels")
            self.config.exemplar_num = self.sample_idx
        except Exception as e:
            print(f"Error, batches::store_effort_labels_dict(): {e}")

    def store_similarity_labels_exemplars_dict(self, anim_name):
        """
        Write out the dictionary of similarity labels and exemplars. Note: self.dict_similarity_exemplars
        contains k,v pairs as -> state_drive : [exemplar1, exemplar2, ...] where each state_drive is an int, and each
        exemplar is of shape (100, 91) and includes PERFORM generated efforts.

        Returns:
            None
        """
        keys = self.dict_similarity_exemplars.keys()
        print(f"storing dict_similarity_exemplars for {anim_name} with len {len(self.dict_similarity_exemplars)},"
              f" key example: {list(keys)[0]}, corresponding value's (list)"
              f" entry's (tensor) shape: {self.dict_similarity_exemplars[next(iter(keys))][0].shape}")

        print("Exemplar lengths for each key:")
        for key, exemplars in self.dict_similarity_exemplars.items():
            lengths = [tensor.shape[0] for tensor in exemplars]
            print(f"Key {key}: {lengths}")

        similarity_dict_path = self.config.similarity_exemplars_dir + anim_name + "_" + self.config.similarity_dict_file_name
        with open(similarity_dict_path, 'wb') as handle:
            pickle.dump(self.dict_similarity_exemplars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"storing {len(self.dict_similarity_exemplars.keys())} similarity class k,v pairs at: {similarity_dict_path}")
        self.dict_similarity_exemplars = self._generate_similarity_classes_exemplars_dict()
        # print(f"reset similarity dict to: {self.dict_similarity_exemplars}")

    def pop_similarity_dict_element(self, key=(0, 0, 0, 0)):
        """
         Remove an element from the similarity dictionary.

         Args:
             key (tuple): The key of the element to remove.

         Returns:
             None
         """
        self.dict_similarity_exemplars.pop(key)

    def move_tuple_to_dict_similarity_front(self, key=(0, 0, 0, 0), dict=None):
        """
        In general, move neutral tuple to front of the similarity dictionary.

        Args:
            key (tuple): The key of the tuple to move.

        Returns:
            None
        """
        if dict is None:
            neut_value = self.dict_similarity_exemplars.pop(key)
            self.dict_similarity_exemplars = {key: neut_value} | self.dict_similarity_exemplars
        else:
            neut_value = dict.pop(key)
            dict_rearranged = {key: neut_value} | dict
            return dict_rearranged


    def convert_exemplar_np_arrays_to_tensors(self):
        """
        Convert exemplar NumPy arrays to TensorFlow tensors.

        Returns:
            None
        """
        for state_drive, inner_list in self.dict_similarity_exemplars.items():
            self.dict_similarity_exemplars[state_drive] = [tf.convert_to_tensor(exemplar, dtype=tf.float32) for
                                                           exemplar in inner_list]
        print(f"converted {len(self.dict_similarity_exemplars[(0, -1, -1, 0)])} similarity exemplars, per class, to type: "
              f"{type(self.dict_similarity_exemplars[(0, -1, -1, 0)][0])}")

    def append_similarity_class_exemplar(self, state_drive, exemplar):
        """
        Append an exemplar into the similarity dict at the corresponding class.
        self.dict_similarity_exemplars contains k,v pairs as -> state_drive : [exemplar1, exemplar2, ...] where each
        exemplar is of shape (100, 91) and includes PERFORM generated efforts. The state_drive is a tuple of effort values.
        The dict item with the most exemplars dictates the number of exemplars for all other classes and therefore the batch
        count for the similarity network.

        Args:
            state_drive (tuple): The similarity class, e.g. (0, -1, -1, 0).
            exemplar (numpy.ndarray): The exemplar to append.

        Returns:
            None
        """
        try:
            # received exemplar of shape: (?, 92) with root joint coordinates or (?, 89) without
            # include efforts but not anim
            # exemplar = np.delete(exemplar, 4, axis=1)

            print(f"append_similarity_class_exemplar: state_drive: {state_drive}, exemplar shape: {exemplar.shape}")
            # ensure appending of exemplar of shape: (100, 91)
            # if exemplar.shape[0] < conf.time_series_size:
            #     exemplar = self.append_to_end_file_exemplar(exemplar)
            # TODO: remove this assertion: AssertionError: Exemplar shape: (30, 88) differs from (30, 91)
            # assert exemplar.shape == conf.similarity_exemplar_dim, (f"Exemplar shape: {exemplar.shape} differs from"
            #                                                         f" {conf.similarity_exemplar_dim}")
            self.dict_similarity_exemplars[state_drive].append(exemplar)
            self.state_drive_exemplar_idx += 1
        except Exception as e:
            print(f"Error batches::append_similarity_class_exemplar(): {e}")

    def verify_dict_similarity_exemplars(self):
        """
        Test function: verify the integrity of the similarity exemplars dictionary.

        Returns:
            None
        """
        print("batches::verify_dict_similarity_exemplars(): Verifying all classes have same number of exemplars")
        list_dict_lens = [len(inner_list) for inner_list in self.dict_similarity_exemplars.values()]
        assert all(dict_len == list_dict_lens[0] for dict_len in list_dict_lens), "Classes differ in exemplar cnt"
        # print("Verifying all exemplars of a class are of same shape, namely shape == conf.similarity_exemplar_dim")
        # for state_drive, inner_list in self.dict_similarity_exemplars.items():
        #     list_exemplar_shapes = [exemplar.shape for exemplar in inner_list]
        #     for exemplar_shape in list_exemplar_shapes:
        #         assert exemplar_shape == conf.similarity_exemplar_dim, \
        #             (f"Exemplar shape: {exemplar_shape} of class {state_drive} differs from:"
        #              f" {conf.similarity_exemplar_dim}")
        # print("Verifying all exemplars are of same shape")
        list_exemplar_shapes = [exemplar.shape for inner_list in self.dict_similarity_exemplars.values() for exemplar
                                in inner_list]
        for exemplar_shape in list_exemplar_shapes:
            if exemplar_shape != list_exemplar_shapes[0]:
                raise AssertionError(f"Exemplar shape {exemplar_shape} differs from: {list_exemplar_shapes[0]}")

    def balance_similarity_classes_by_exemplar_count(self):
        """
        Balance the similarity classes by extending them with exemplars.

        Returns:
            None
        """
        print(f"batches::balance_similarity_classes() called ...")
        inner_lists = [[inner_list, len(inner_list)] for inner_list in self.dict_similarity_exemplars.values()]
        # get max inner list by second element of inner list
        max_inner_list = max(inner_lists, key=lambda x: x[1])
        max_inner_list = max_inner_list[0]
        # max_inner_list = max(inner_list for inner_list in self.dict_similarity_exemplars.values())
        tmp_exemplar = max_inner_list[0]
        len_max_inner_list = len(max_inner_list)
        print(f"balance_similarity_classes: len_max_inner_list: {len_max_inner_list}")
        inner_list_lens = []
        for state_drive, inner_list in self.dict_similarity_exemplars.items():
            length_inner_list = len(inner_list)
            # inner_list_lens.append(length_inner_list)
            # num_inner_list_elems = length_inner_list
            # for toy data lacking exemplars to multiple classes
            if length_inner_list == 0:
                length_inner_list = 1
                # {exemplar idx: exemplar}
                inner_list.append(tmp_exemplar)
            # convert num_inner_list_elems to len_max_inner_list
            if length_inner_list < len_max_inner_list:
                # perform pop of inner_list
                last_exemplar = inner_list[-1]
                for i in range(length_inner_list, len_max_inner_list):
                    # new_exemplar = self.append_to_end_file_exemplar(last_exemplar, make_whole_exemplar=True)
                    # inner_list.append(new_exemplar)
                    inner_list.append(last_exemplar)
                # print(f"Newly appended similarity class exemplar list is of len: {len(inner_list)}")
            inner_list_lens.append(len(inner_list))
        assert all(dict_len == inner_list_lens[0] for dict_len in inner_list_lens), "Classes have different exemplar " \
                                                                                    "counts"

    def balance_single_exemplar_similarity_classes_by_frame_count(self, anim_name):
        """
        Balance the similarity classes by extending them with exemplars.

        Returns:
            None
        """
        # Get the maximum frame count across all exemplars
        max_frame_count = max(
            len(exemplar) for inner_list in self.dict_similarity_exemplars.values() for exemplar in inner_list)

        cached_frame_counts = []
        for state_drive, value_list in self.dict_similarity_exemplars.items():
            for i in range(len(value_list)):
                lone_exemplar = value_list[i]
                cached_frame_counts.append(len(lone_exemplar))
                # If the exemplar's frame count is less than the max frame count, extend it
                if len(lone_exemplar) < max_frame_count:
                    last_frame = lone_exemplar[-1]
                    additional_frames = np.repeat(last_frame[np.newaxis, :], max_frame_count - len(lone_exemplar), axis=0)
                    value_list[i] = np.concatenate((lone_exemplar, additional_frames), axis=0)

        # Verify that all exemplars now have the same frame count
        count=0
        for value_list in self.dict_similarity_exemplars.values():
            count += 1
            for lone_exemplar in value_list:
                assert len(lone_exemplar) == max_frame_count, f"Exemplar {count} frame count {len(lone_exemplar)} does not match max frame count {max_frame_count}"
        print(f"Anim {anim_name} max frame count: {max_frame_count} ... distribution of frame counts prior padding: {cached_frame_counts}")

    @staticmethod
    def _generate_similarity_classes_exemplars_dict():
        """
        Generate Dict for mapping Similarity class to exemplar number and data
        Returns:
            dict_similarity_exemplars: Dictionary. {similarity_class: []}

        Once populated, using training dataset: stylized motion effort_walking_105_34_552.bvh, the dict will look like:
        {(0, 0, 0, 0): [tensor([100, 91]), tensor([100, 91]), ... ], (0, 0, 0, 1): ...} for 57 keys, each with
        a list of 37 exemplars of size (100, 91). This makes each class span the length of about two minutes and the
        dictionary size of 2 GB.
        """

        def generate_states_and_drives():
            """
            Create list of all state and drive combinations

            Returns:
                None
            """
            def convert_to_nary(tuple_index, values_per_effort, efforts_per_tuple):
                effort_tuple = []
                zeroes_counter = 0
                for _ in range(efforts_per_tuple):
                    tuple_index, remainder = divmod(tuple_index, values_per_effort)
                    if remainder == 1:
                        zeroes_counter += 1
                    effort_tuple.append(effort_vals[remainder])
                if zeroes_counter == 2 or zeroes_counter == 1:
                    # print(f"polarized stat or drive found: {effort_tuple}")
                    states_and_drives.append(effort_tuple)

            states_and_drives = []
            effort_vals = [-1, 0, 1]
            values_per_effort = 3
            efforts_per_tuple = 4
            tuple_index = 0
            while tuple_index < math.pow(values_per_effort, 4):
                tuple_index += 1
                convert_to_nary(tuple_index, values_per_effort, efforts_per_tuple)
            # include the neutral tuple
            states_and_drives.append([0, 0, 0, 0])
            return states_and_drives

        states_and_drives = generate_states_and_drives()
        return {tuple(state_drive): [] for state_drive in states_and_drives}
