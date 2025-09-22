"""
static module for organizing synthetic motion data in the context of both efforts and similarity networks
"""
import sys

import torch
import tensorflow as tf
from src.batches import Batches
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from sklearn.preprocessing import StandardScaler
from pymo.preprocessing import *
from scipy.spatial.transform import Rotation
from os import path
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import warnings

anim_ind = {'WALKING': 0, 'POINTING': 1, 'PICKING': 2, 'WAVING': 3, 'THROWING': 4, 'AIMING': 5, 'JUMPING': 6,
            'RUNNING': 7}
parser = BVHParser()


def visualize(file_bvh):
    """
    Visualize motion data from a BVH file.

    Args:
        file_bvh (str): The path to the BVH file to visualize.

    Returns:
        None
    """
    parsed_data = parser.parse(file_bvh)
    data_pipe = Pipeline([
        ('param1', MocapParameterizer('expmap')),
    ])
    data = parsed_data
    mp = MocapParameterizer('position')
    positions = mp.transform([data])[0]

    nb_play_mocap(positions, 'pos',
                  scale=2, camera_z=800, frame_time=1 / 30,
                  base_url='pymo/mocapplayer/playBuffer2.html')


def clear_file(file):
    """
    Remove character name from a text file.

    Args:
        file (str): The path to the text file.

    Returns:
        None
    """
    # removes character name from the file
    # read input file
    fin = open(file, "rt")
    # read file contents to string
    data = fin.read()
    # replace all occurrences of the required string
    data = data.replace('Carl:', '')
    # close the input file
    fin.close()
    # open the input file in write mode
    fin = open(file, "wt")
    # overwrite the input file with the resulting data
    fin.write(data)
    # close the file
    fin.close()


def prep_all_data_for_training(config_instance, batches_instance, rotations=True, velocities=False, similarity_pre_processing_only=True, anim_name=None):
    """
    Prepare motion data for training.

    Args:
        config_instance: Configuration instance
        batches_instance: Batches instance for storing processed data
        rotations (bool): Whether to include rotation values in exemplars
        velocities (bool): Whether to include velocities in exemplars
        similarity_pre_processing_only (bool): If True, only process data for similarity network
        anim_name (str): Animation name (walking, pointing, or picking)

    Returns:
        None
    """

    # def _preprocess_pipeline(parsed_data):
    #     """
    #     Process BVH motion data by converting Euler rotations to quaternions.
    #
    #     Args:
    #         parsed_data: Parsed BVH data (pymo.data.MocapData instance)
    #
    #     Returns:
    #         numpy.ndarray: Motion data with rotations represented as quaternions
    #     """
    #     print(f"Processing BVH data with {len(parsed_data.skeleton)} joints, {parsed_data.values.shape[0]} frames")
    #
    #     # First convert to numpy array with Euler angles
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         # Use 'euler' as param type to get Euler angles
    #         data_pipe_euler = Pipeline(steps=[
    #             ('param', MocapParameterizer('euler')),
    #             ('np', Numpyfier()),
    #         ])
    #
    #     euler_data = data_pipe_euler.fit_transform([parsed_data])[0]
    #     print(f"Euler data shape: {euler_data.shape}")
    #
    #     # Separate position and rotation data
    #     # First 3 columns are root positions (x, y, z)
    #     positions = euler_data[:, :3]  # Root joint positions
    #     euler_rotations = euler_data[:, 3:]  # All joint rotations
    #
    #     # Convert Euler rotations to quaternions
    #     # Reshape to group by joints (each joint has 3 rotations: z, x, y according to BVH CHANNELS)
    #     n_frames = euler_rotations.shape[0]
    #     n_rotation_values = euler_rotations.shape[1]
    #     n_joints = n_rotation_values // 3
    #
    #     print(f"Number of frames: {n_frames}")
    #     print(f"Number of joints with rotations: {n_joints}")
    #
    #     euler_rotations_reshaped = euler_rotations.reshape(n_frames, n_joints, 3)
    #
    #     # Initialize array for quaternions (4 values per quaternion: x, y, z, w)
    #     quat_rotations = np.zeros((n_frames, n_joints * 4))
    #
    #     # Convert each joint's Euler angles to quaternion
    #     # BVH typically uses 'ZXY' order as per the provided BVH structure
    #     print("Converting Euler angles to quaternions...")
    #     for frame in range(n_frames):
    #         for joint in range(n_joints):
    #             # Get Euler angles for this joint
    #             euler_angles = euler_rotations_reshaped[frame, joint]
    #
    #             # Convert to quaternion using 'zxy' order
    #             rot = Rotation.from_euler('zxy', euler_angles, degrees=True)
    #             quat = rot.as_quat()  # [x, y, z, w] format
    #
    #             # Store in output array
    #             quat_rotations[frame, joint * 4:(joint + 1) * 4] = quat
    #
    #     print(f"Quaternion rotations shape: {quat_rotations.shape}")
    #
    #     # Combine positions and quaternion rotations
    #     combined_data = np.hstack((positions, quat_rotations))
    #     print(f"Final combined data shape: {combined_data.shape}")
    #
    #     # Show the expansion in data size
    #     print(f"Original Euler data had {euler_data.size} values")
    #     print(f"Quaternion data has {combined_data.size} values")
    #     print(f"Ratio rotations (quat/euler): {combined_data.size / euler_rotations.size:.2f}")
    #
    #     return combined_data

    def _preprocess_pipeline(parsed_data):
        """
        Process BVH motion data by converting Euler rotations to normalized quaternions.

        Args:
            parsed_data: Parsed BVH data (pymo.data.MocapData instance)

        Returns:
            numpy.ndarray: Motion data with normalized quaternions
        """
        print(f"Processing BVH data with {len(parsed_data.skeleton)} joints, {parsed_data.values.shape[0]} frames")

        # First convert to numpy array with Euler angles
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_pipe_euler = Pipeline(steps=[
                ('param', MocapParameterizer('euler')),
                ('np', Numpyfier()),
            ])

        euler_data = data_pipe_euler.fit_transform([parsed_data])[0]
        print(f"Euler data shape: {euler_data.shape}")

        # Separate position and rotation data
        positions = euler_data[:, :3]  # Root joint positions
        euler_rotations = euler_data[:, 3:]  # All joint rotations

        # Convert Euler rotations to quaternions
        n_frames = euler_rotations.shape[0]
        n_rotation_values = euler_rotations.shape[1]
        n_joints = n_rotation_values // 3

        print(f"Number of frames: {n_frames}")
        print(f"Number of joints with rotations: {n_joints}")

        euler_rotations_reshaped = euler_rotations.reshape(n_frames, n_joints, 3)

        # Initialize array for quaternions
        quat_rotations = np.zeros((n_frames, n_joints * 4))

        print("Converting and normalizing Euler angles to quaternions...")
        for frame in range(n_frames):
            for joint in range(n_joints):
                # Convert to quaternion
                rot = Rotation.from_euler('zxy', euler_rotations_reshaped[frame, joint], degrees=True)
                quat = rot.as_quat()  # [x, y, z, w]

                # Normalize quaternion
                quat /= np.linalg.norm(quat)

                # Store normalized quaternion
                quat_rotations[frame, joint * 4:(joint + 1) * 4] = quat

        print(f"Quaternion rotations shape: {quat_rotations.shape}")

        # Combine positions and quaternion rotations
        combined_data = np.hstack((positions, quat_rotations))
        print(f"Final combined data shape: {combined_data.shape}")

        return combined_data

    def _get_standardized_rotations(data_expmaps):
        data_expmaps = _z_score_generator(data_expmaps)
        return data_expmaps

    def _get_standardized_velocities(data_velocities):
        # needed for proper broadcasting of following step
        frame_rate_array = np.tile(bvh_frame_rate.pop(), (data_velocities.shape[0] - 1, data_velocities.shape[1]))
        # calculate velocities from positions
        data_velocities[1:] = (data_velocities[1:, :] - data_velocities[:-1, :]) / frame_rate_array
        data_velocities[0] = 0
        # standardize velocities
        data_velocities = _z_score_generator(data_velocities)
        return data_velocities

    def _z_score_generator(np_array):
        scaler = StandardScaler()
        scaler = scaler.fit(np_array)
        np_array = scaler.transform(np_array)
        return np_array

    # def create_corr_matrix(np_array, name):
    #     print("NAME:", name)
    #     dataframe = pd.DataFrame(np_array)
    #     print("DATAFRAME :", dataframe)
    #     a = dataframe.corr()
    #     ax = sns.heatmap(
    #         a,
    #         vmin=-1, vmax=1, center=0,
    #         cmap=sns.diverging_palette(20, 220, n=200),
    #         square=True
    #     )
    #     plt.title(name)
    #     plt.show()

    def apply_moving_window(batches, file_data):
        """
        nested function of prep_all_data_for_training()

        handles both construction of effort network batches (with rotations only, each batch has
         dim = batch_size x time_series_size x 87) and similarity network class to exemplar dict.

        Args:
            batches: instance of Batches class
            file_data: np.array comprised of preprocessed motion data + effort values + anim name

        Returns:
            None
        """
        print(f"osd::apply_moving_window(): {anim_name} ... Applying moving window to file data")
        start_index = config_instance.time_series_size
        end_index = file_data.shape[0]
        for i in range(start_index, end_index + config_instance.window_delta, config_instance.window_delta):
            indices = range(i - config_instance.time_series_size, i)
            # end of file corner case correction
            if i > end_index:
                indices = range(i - config_instance.time_series_size, end_index)
                exemplar = file_data[indices]
                exemplar = batches.append_to_end_file_exemplar(exemplar)
                batches.append_efforts_batch_and_labels(exemplar)
                if f == filenames[-1]:
                    batches.extend_final_batch(exemplar)
            else:
                exemplar = file_data[indices]
                batches.append_efforts_batch_and_labels(exemplar)
            if tuple_effort_list in batches.dict_similarity_exemplars.keys():
                # storing similarity class exemplar here while iterating through the file and storing effort batches
                batches.append_similarity_class_exemplar(tuple_effort_list, file_data[indices])
            if len(batches.current_batch_exemplar[batches.batch_idx]) == config_instance.batch_size_efforts_network:
                batches.store_efforts_batch()

    try:
        singleton_batches = batches_instance
        bvh_counter = 0
        bvh_frame_rate = set()

        # Set directory based on animation name
        if anim_name == "walking":
            dir_filenames = config_instance.bvh_files_dir_walking
            filenames = os.listdir(dir_filenames)
        elif anim_name == "pointing":
            dir_filenames = config_instance.bvh_files_dir_pointing
            filenames = os.listdir(dir_filenames)
        elif anim_name == "picking":
            dir_filenames = config_instance.bvh_files_dir_picking
            filenames = os.listdir(dir_filenames)
        else:
            raise ValueError("anim_name must be one of the following: WALKING, POINTING, PICKING")

        print(
            f"osd::prep_all_data_for_training(): {anim_name} filenames dir: {dir_filenames}, num files: {len(filenames)}")

        # Import needed here to avoid circular imports
        from pymo.parsers import BVHParser
        parser = BVHParser()

        for f in filenames:
            if f.endswith("bvh"):
                print(f"path: {f}")
                name = os.path.splitext(f)[0]  # exclude extension bvh by returning the root
                name_split = name.split('_')  # get effort values from the file name
                print(f"name_split: {name_split}, length: {len(name_split)}")
                anim = name_split[0]
                f_full_path = os.path.join(dir_filenames, f)
                print(f"Processing file: {f_full_path}")

                # Extract effort values from filename
                efforts_list = [float(p) for p in name.split('_')[-4:]]
                tuple_effort_list = tuple(efforts_list)

                # Skip if not in target exemplars
                if similarity_pre_processing_only:
                    if tuple_effort_list not in singleton_batches.dict_similarity_exemplars.keys():
                        continue

                singleton_batches.state_drive_exemplar_idx = 0

                # Parse BVH file
                parsed_data = parser.parse(f_full_path)
                bvh_frame_rate.add(parsed_data.framerate)
                assert len(bvh_frame_rate) == 1, f"More than one frame rate present!!! {bvh_frame_rate}"

                # Process the data based on configuration
                if rotations and velocities:
                    # Process with both rotations and velocities
                    data_quats = _preprocess_pipeline(parsed_data)
                    # Remove root joint absolute positions if needed
                    data_quats = data_quats[:, 3:]
                    # data_velocities = _get_standardized_velocities(data_quats)
                    # Stack quaternion rotations and velocities
                    data = np.hstack((data_quats, data_quats))
                elif not rotations and velocities:
                    # Process with only velocities
                    data_quats = _preprocess_pipeline(parsed_data)
                    # data = _get_standardized_velocities(data_quats)
                else:
                    # Process with only rotations (default)
                    data_quats = _preprocess_pipeline(parsed_data)
                    print(f"Processed motion matrix of shape: {data_quats.shape}")
                    # Remove root joint absolute positions
                    data_quats = data_quats[:, 3:]
                    print(f"Motion matrix after removing root positions: {data_quats.shape}")
                    # data = _get_standardized_rotations(data_quats)

                bvh_counter += 1

                # Check if data is large enough
                if data_quats.shape[0] < config_instance.time_series_size:
                    assert False, f"Preprocessed file too small- {data_quats.shape[0]} - relative to exemplar size -" \
                                  f" {config_instance.time_series_size}"

                file_data = data_quats
                # # Add effort values and animation type to data
                # f_rep = np.tile(efforts_list, (data.shape[0], 1))
                # # Add animation type as an additional column
                # anim_ind = {"WALKING": 0, "POINTING": 1, "PICKING": 2}  # Map animation names to indices
                # a_rep = np.tile(anim_ind[str.upper(anim)], (data.shape[0], 1))
                #
                # # Combine data
                # file_data = np.concatenate((a_rep, data), axis=1)
                # file_data = np.concatenate((f_rep, file_data), axis=1)
                #
                # print(f"Motion matrix after adding effort values and anim name: {file_data.shape}")

                # Store or process the data
                if similarity_pre_processing_only:
                    print(
                        f"Anim: {anim_name}, {bvh_counter} ... Appending similarity class exemplar for tuple: {tuple_effort_list}")
                    singleton_batches.append_similarity_class_exemplar(tuple_effort_list, file_data)
                else:
                    apply_moving_window(singleton_batches, file_data)

        # Post-processing steps
        config_instance.bvh_file_num = bvh_counter
        # singleton_batches.balance_single_exemplar_similarity_classes_by_frame_count(anim_name)
        singleton_batches.move_tuple_to_dict_similarity_front(key=(0, 0, 0, 0))
        singleton_batches.convert_exemplar_np_arrays_to_tensors()
        singleton_batches.store_similarity_labels_exemplars_dict(anim_name)

        # Verify data integrity
        assert singleton_batches.batch_idx == len(
            singleton_batches.dict_efforts_labels.values()) - 1, f"batch_idx: {singleton_batches.batch_idx}, " \
                                                                 f"num" \
                                                                 f"labels: {len(singleton_batches.dict_efforts_labels.values())}"
        # singleton_batches.verify_dict_similarity_exemplars()
    except Exception as e:
        print(f"Error in prep_all_data_for_training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit()


def load_similarity_data(bool_drop, anim_name, config, train_val_split=1):
    """
    Load similarity dict of all class exemplars and split across train, validation, and test sets.

    Args:
        train_val_split: float:keep at 1.0; vestigial param given that splitting occurs after returning to run_motion_triplet_training.py

    Returns:
        similarity_dict: dict: partitioned similarity dict of all class exemplars
    """

    file_path = config.similarity_exemplars_dir + anim_name + "_" + config.similarity_dict_file_name
    singleton_batches = Batches(config)
    if not os.path.isfile(file_path):
        print(f"osd::load_similarity_data(): Generating similarity data for {anim_name} with path: {file_path}")
        prep_all_data_for_training(config_instance=config, batches_instance=singleton_batches, rotations=True,
                                   velocities=False, similarity_pre_processing_only=True,
                                   anim_name=anim_name)

    # Load the dictionary
    dict_similarity_classes_exemplars = pickle.load(open(file_path, "rb"))

    # Print information about the loaded dictionary structure
    # print(f"\nDICTIONARY STRUCTURE EXPLORATION FOR {anim_name}:")
    # print(f"Number of keys in dictionary: {len(dict_similarity_classes_exemplars)}")

    # Check 2-3 example keys and their values
    sample_keys = list(dict_similarity_classes_exemplars.keys())[:3]  # Take first 3 keys for example
    # print(f"Sample keys: {sample_keys}")

    # Explore the nested structure for each sample key
    for idx, key in enumerate(sample_keys):
        exemplars = dict_similarity_classes_exemplars[key]
        # print(f"\nKey {idx + 1}: {key}")
        # print(f"  Number of exemplars: {len(exemplars)}")

        if len(exemplars) > 0:
            # Check the type and shape of exemplars
            exemplar = exemplars[0]
            # print(f"  First exemplar type: {type(exemplar)}")

            if isinstance(exemplar, (torch.Tensor, np.ndarray, tf.Tensor)):
                if isinstance(exemplar, torch.Tensor):
                    shape = exemplar.shape
                    dtype = exemplar.dtype
                elif isinstance(exemplar, np.ndarray):
                    shape = exemplar.shape
                    dtype = exemplar.dtype
                elif isinstance(exemplar, tf.Tensor):
                    shape = exemplar.shape
                    dtype = exemplar.dtype
                # print(f"  First exemplar shape: {shape}")
                # print(f"  First exemplar dtype: {dtype}")
            else:
                print(f"  First exemplar is not a tensor or array, it's: {type(exemplar)}")

    # Check if all exemplars have the same length (first dimension)
    lengths = []
    for key in dict_similarity_classes_exemplars:
        if dict_similarity_classes_exemplars[key]:  # If there are exemplars
            exemplar = dict_similarity_classes_exemplars[key][0]
            if hasattr(exemplar, 'shape'):
                lengths.append((key, exemplar.shape[0]))

    # print("\nSequence lengths:")
    # # Print first 5 lengths for brevity
    # for key, length in lengths[:5]:
    #     print(f"  Key {key}: Length {length}")

    # Check if all lengths are the same
    unique_lengths = set(length for _, length in lengths)
    print(f"Number of unique lengths: {len(unique_lengths)}")
    if len(unique_lengths) <= 3:  # If there are only a few unique lengths, print them all
        print(f"Unique lengths: {unique_lengths}")
    else:
        print(f"Range of lengths: {min(unique_lengths)} to {max(unique_lengths)}")

    # Keys (sample): [(0, 0, 0, 0), (0, -1, -1, -1), (-1, 0, -1, -1), (0, 0, -1, -1), (1, 0, -1, -1)]
    # where each value is a list of a single numpy array (e.g, shape: (137, 88)) and all such tensors have been made uniform in their frame count
    #TODO: verif that we are indeed storing numpy arrays as the payload. And, given how motion units as opposed to snippets, eliminate the list use
    print(f"loaded dict_similarity_classes_exemplars for anim {anim_name}")

    if bool_drop:
        config.similarity_per_anim_class_num = 56
        dict_similarity_classes_exemplars.pop((0, 0, 0, 0))
    else:
        config.similarity_per_anim_class_num = 57
        # ensure element of key (0, 0, 0, 0) is at the front of the dict
        print(f"Anim: {anim_name}, moving dict entry for key (0,0,0,0) to front of dict")
        dict_similarity_classes_exemplars = singleton_batches.move_tuple_to_dict_similarity_front(key=(0, 0, 0, 0), dict=dict_similarity_classes_exemplars)
    # singleton_batches.dict_similarity_exemplars = dict_similarity_classes_exemplars
    # next(iter(dict_similarity_classes_exemplars.keys())) gets the first key in the dictionary
    # the length of the lone entry of the value (itself a list) somehow specifies the number of exemplars
    num_exemplars = len(dict_similarity_classes_exemplars[next(iter(dict_similarity_classes_exemplars.keys()))])
    print(f"{anim_name}: Number of total classes: {len(dict_similarity_classes_exemplars)}")
    print(f"{anim_name}: Number of total exemplars per class: {num_exemplars}")
    # print(f"{anim_name}: Frame count for first exemplar: {len(dict_similarity_classes_exemplars[(0, 0, 0, 0)][0])}")
    # print(f"{anim_name}: Shape for first exemplar: {(dict_similarity_classes_exemplars[(0, 0, 0, 0)][0].shape)}")
    p = np.random.permutation(num_exemplars - 1)
    train_size = int(train_val_split * num_exemplars)
    # temp change to inc val set size
    # val_and_test_size = int(((1 - train_val_split) * num_exemplars) / 2)
    val_and_test_size = int(((1 - train_val_split) * num_exemplars))
    print(f"train size: {train_size}, val and test size: {val_and_test_size}")

    train_data = {}
    validation_data = {}
    test_data = {}
    for k, v in dict_similarity_classes_exemplars.items():
        train_data[k] = v[:train_size]
        if val_and_test_size == 0:
            validation_data[k] = v[:train_size]
            test_data[k] = v[:train_size]
        else:
            validation_data[k] = v[train_size:train_size + val_and_test_size]
            test_data[k] = v[train_size:train_size + val_and_test_size]

    return {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }


def balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts, max_frame_count):
    """
    Balance the number of frames in each class exemplar to the same number of frames as the class exemplar with the
    most frames.

    Args:
        None

    Returns:
        None
    """
    balanced_dicts = []
    # Get the maximum frame count across all exemplars in all dictionaries
    # max_frame_count = max(
    #     len(exemplar) for dict_similarity_exemplars in list_similarity_dicts for inner_list in
    #     dict_similarity_exemplars.values() for exemplar in inner_list)
    print(f"balance_exemplar_similarity_classes_by_frame_count: max_frame_count: {max_frame_count}")

    for dict_similarity_exemplars in list_similarity_dicts:
        for state_drive, inner_list in dict_similarity_exemplars.items():
            count_exemplars = 0
            for i in range(len(inner_list)):
                exemplar = inner_list[i]
                count_exemplars += 1
                # print(
                #     f"balance_exemplar_similarity_classes_by_frame_count: state_drive: {state_drive}, exemplar count {count_exemplars} shape: {exemplar.shape}")
                # Note that exemplar is of type tensorflow.python.framework.ops.EagerTensor but gets represented as a numpy array after extending it
                if len(exemplar) < max_frame_count:
                    last_frame = exemplar[-1]
                    additional_frames = np.repeat(last_frame[np.newaxis, :], max_frame_count - len(exemplar),
                                                  axis=0)
                    inner_list[i] = np.concatenate((exemplar, additional_frames), axis=0)
                # print(
                #     f"balance_exemplar_similarity_classes_by_frame_count: state_drive: {state_drive}, exemplar count {count_exemplars} final shape: {inner_list[i].shape}")

        # Verify that all exemplars now have the same frame count
        count = 0
        for inner_list in dict_similarity_exemplars.values():
            count += 1
            for exemplar in inner_list:
                assert len(
                    exemplar) == max_frame_count, f"Exemplar {count} frame count {len(exemplar)} does not match max frame count {max_frame_count}"

        balanced_dicts.append(dict_similarity_exemplars)

    return balanced_dicts


"""
"""
"""

Modified functions for organize_synthetic_data.py to work with embeddings

"""
"""
"""


def load_similarity_data_from_embeddings(bool_drop, anim_name, config, embedding_dir, combination_method='concat',
                                         train_val_split=1, force_regenerate=False):
    """
    Load similarity data from pre-generated embeddings instead of raw motion data.

    Args:
        bool_drop: Whether to drop neutral exemplar
        anim_name: Animation name (should be "walking" for your case)
        config: Configuration object
        embedding_dir: Directory containing the embedding files
        combination_method: How to combine root and rotation embeddings
        train_val_split: Train/validation split ratio
        force_regenerate: If True, force regeneration even if pickle file exists

    Returns:
        similarity_dict: Dictionary with train/validation/test splits
    """
    from embedding_dataset import EmbeddingDataset  # Import the class we created above

    # Create expected file path for the similarity dictionary
    file_path = config.similarity_exemplars_dir + anim_name + "_" + config.similarity_dict_file_name
    embedding_file_path = config.similarity_exemplars_dir + f"{anim_name}_embeddings_{combination_method}_" + config.similarity_dict_file_name

    # Check if embedding-based similarity data already exists
    if not os.path.isfile(embedding_file_path) or force_regenerate:
        print(f"Generating fresh similarity data from embeddings for {anim_name}")
        print(f"Embedding method: {combination_method}")

        # Create embedding dataset
        embedding_dataset = EmbeddingDataset(embedding_dir)

        # Create similarity dictionary from embeddings
        dict_similarity_classes_exemplars = embedding_dataset.create_similarity_dict(combination_method)

        # Save the dictionary with a new name to distinguish from raw motion data
        os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
        with open(embedding_file_path, 'wb') as f:
            pickle.dump(dict_similarity_classes_exemplars, f)
        print(f"Saved embedding-based similarity data to {embedding_file_path}")
    else:
        # Load existing embedding-based similarity data
        dict_similarity_classes_exemplars = pickle.load(open(embedding_file_path, "rb"))
        print(f"Loaded existing embedding-based similarity data from {embedding_file_path}")

    # Print information about the loaded data
    print(f"\nEMBEDDING-BASED SIMILARITY DATA FOR {anim_name}:")
    print(f"Number of effort classes: {len(dict_similarity_classes_exemplars)}")

    # Check a few example classes
    sample_keys = list(dict_similarity_classes_exemplars.keys())[:3]
    for key in sample_keys:
        exemplars = dict_similarity_classes_exemplars[key]
        if exemplars:
            print(f"  Effort {key}: {len(exemplars)} exemplars, embedding shape: {exemplars[0].shape}")

    # Handle neutral class dropping
    if bool_drop:
        config.similarity_per_anim_class_num = len(dict_similarity_classes_exemplars) - 1
        if (0, 0, 0, 0) in dict_similarity_classes_exemplars:
            dict_similarity_classes_exemplars.pop((0, 0, 0, 0))
    else:
        config.similarity_per_anim_class_num = len(dict_similarity_classes_exemplars)
        # Move neutral to front if it exists
        if (0, 0, 0, 0) in dict_similarity_classes_exemplars:
            neutral_value = dict_similarity_classes_exemplars.pop((0, 0, 0, 0))
            dict_similarity_classes_exemplars = {(0, 0, 0, 0): neutral_value, **dict_similarity_classes_exemplars}

    # Get number of exemplars per class
    if dict_similarity_classes_exemplars:
        num_exemplars = len(dict_similarity_classes_exemplars[next(iter(dict_similarity_classes_exemplars.keys()))])
        print(f"{anim_name}: Number of exemplars per class: {num_exemplars}")
        print(
            f"{anim_name}: Embedding dimension: {dict_similarity_classes_exemplars[next(iter(dict_similarity_classes_exemplars.keys()))][0].shape}")
    else:
        num_exemplars = 0

    # Create train/validation/test splits
    train_size = int(train_val_split * num_exemplars)
    val_and_test_size = int(((1 - train_val_split) * num_exemplars))

    train_data = {}
    validation_data = {}
    test_data = {}

    for k, v in dict_similarity_classes_exemplars.items():
        train_data[k] = v[:train_size] if train_size > 0 else v
        if val_and_test_size == 0:
            validation_data[k] = v[:train_size] if train_size > 0 else v
            test_data[k] = v[:train_size] if train_size > 0 else v
        else:
            validation_data[k] = v[train_size:train_size + val_and_test_size]
            test_data[k] = v[train_size:train_size + val_and_test_size]

    return {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }


def balance_embedding_similarity_classes(list_similarity_dicts):
    """
    Balance embedding similarity classes. Since embeddings have fixed dimensions,
    we just need to ensure all classes have the same number of exemplars.

    Args:
        list_similarity_dicts: List of similarity dictionaries

    Returns:
        Balanced list of dictionaries
    """
    balanced_dicts = []

    for dict_similarity_exemplars in list_similarity_dicts:
        # Find the minimum number of exemplars across all classes
        min_exemplars = min(len(exemplars) for exemplars in dict_similarity_exemplars.values())
        print(f"Balancing to {min_exemplars} exemplars per class")

        # Truncate all classes to have the same number of exemplars
        for key, exemplars in dict_similarity_exemplars.items():
            dict_similarity_exemplars[key] = exemplars[:min_exemplars]

        balanced_dicts.append(dict_similarity_exemplars)

    return balanced_dicts


# Alternative: Create a modified SimilarityDataLoader for embeddings
class EmbeddingSimilarityDataLoader:
    """
    Modified data loader that works directly with embeddings instead of motion data.
    """

    def __init__(self, list_similarity_dicts, config, shuffle=False, valid_indices=None):
        """
        Initialize data loader for embeddings.

        Args:
            list_similarity_dicts: List of similarity dictionaries containing embeddings
            config: Configuration object
            shuffle: Whether to shuffle data
            valid_indices: Valid indices for train/val splitting
        """
        self.config = config
        self.shuffle = shuffle
        self.dict_similarity_exemplars = {}
        self.class_indexes = []
        self.list_tuples_dict_idx_class_tuple = []

        # Track module sizes and start indices
        self.module_sizes = []
        self.module_start_indices = []

        all_classes_count = 0
        start_idx = 0

        for i, similarity_dict in enumerate(list_similarity_dicts):
            curr_valid_indices = None if valid_indices is None else valid_indices[i]

            # Count examples for this module
            module_examples = 0
            for class_tuple in similarity_dict.keys():
                if curr_valid_indices is None or class_tuple in curr_valid_indices:
                    module_examples += 1

            self.module_sizes.append(module_examples)
            self.module_start_indices.append(start_idx)
            start_idx += module_examples

            # Add examples to our data structure
            for class_tuple, embeddings in similarity_dict.items():
                if curr_valid_indices is not None and class_tuple not in curr_valid_indices:
                    continue

                new_key = (i, class_tuple)
                # Take first embedding as the representative (since embeddings are already computed)
                self.dict_similarity_exemplars[new_key] = embeddings[0] if embeddings else np.zeros(
                    config.embedding_refinement_model_output_size)
                self.list_tuples_dict_idx_class_tuple.append(new_key)
                self.class_indexes.append(all_classes_count)
                all_classes_count += 1

        self.num_classes = len(self.class_indexes)
        self.batch_size = len(self.dict_similarity_exemplars.keys())
        self._num_batches = 1
        self.exemplar_idx = 0

        # Set exemplar dimensions based on embedding size
        if self.dict_similarity_exemplars:
            first_embedding = next(iter(self.dict_similarity_exemplars.values()))
            if hasattr(first_embedding, 'shape'):
                self.exemplar_dim = first_embedding.shape
            else:
                self.exemplar_dim = (len(first_embedding),)
        else:
            self.exemplar_dim = (config.embedding_refinement_model_output_size,)

        print(f"EmbeddingSimilarityDataLoader: {self.num_classes} classes, embedding dim: {self.exemplar_dim}")

    def __getitem__(self, index):
        """Get a batch of embeddings."""
        if index >= self._num_batches:
            raise StopIteration

        # Stack all embeddings into a batch
        batch_embeddings = []
        for class_tuple in self.list_tuples_dict_idx_class_tuple:
            embedding = self.dict_similarity_exemplars[class_tuple]
            if isinstance(embedding, np.ndarray):
                batch_embeddings.append(torch.from_numpy(embedding).float())
            else:
                batch_embeddings.append(embedding.float())

        batch_features = torch.stack(batch_embeddings)

        # Add dummy dimensions to match expected format (if needed)
        if len(batch_features.shape) == 2:  # [batch_size, embedding_dim]
            batch_features = batch_features.unsqueeze(-1)  # [batch_size, embedding_dim, 1]

        class_labels = torch.tensor(self.class_indexes, dtype=torch.long)

        return batch_features, class_labels

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self._num_batches:
            raise StopIteration

        batch = self.__getitem__(self.current_index)
        self.current_index += 1
        return batch

    def on_epoch_end(self):
        """Handle end of epoch."""
        if self.shuffle:
            p = np.random.permutation(len(self.class_indexes))
            self.class_indexes = [self.class_indexes[i] for i in p]
            self.list_tuples_dict_idx_class_tuple = [self.list_tuples_dict_idx_class_tuple[i] for i in p]
        self.exemplar_idx = 0

