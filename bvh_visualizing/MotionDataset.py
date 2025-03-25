import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from bvh_visualizing.bvh import BVH
import pandas as pd


class MotionDataset(Dataset):
    def __init__(self, csv_file, data_folder):
        """
        Args:
            csv_file (str): Path to the 'ratios<Motion>.csv' file.
            data_folder (str): Path to the folder containing motion files.
        """
        self.data_folder = data_folder


        # Read the CSV file
        self.triplet_data =  pd.read_csv(csv_file)


        self.bvh_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".bvh")]



    def efforts_to_filename(self, motion_name, effort_vector):
        """
        Converts an effort vector to a motion filename.
        For example, effort_vector [1, 1, 1, 1] becomes 'motionName_1_1_1_1'.
        """
        effort_str = '_'.join(map(str, effort_vector))
        filename = f"{motion_name}_{effort_str}.bvh"
        return filename

    def load_motion(self, motion_name, effort_vector):
        """
        Loads motion data for a given effort vector from the dataset folder.
        """
        filename = self.efforts_to_filename(motion_name, effort_vector)
        bvh_file = os.path.join(self.data_folder, filename)
        if not os.path.exists(bvh_file):
            raise FileNotFoundError(f"Motion file {bvh_file} not found.")


        animation = BVH()
        animation.load(bvh_file)

        motion = self.extract_root_and_rotations(animation)
        # normalized_rotations = self.normalize_rotations(rotations)

        # return rotations #normalized_rotations
        return motion



    def get_input_dim(self, vec_dim):
        #Assuming all the inputs have the same feature size
        bvh_file = self.bvh_files[0]
        bvh = BVH()
        bvh.load(bvh_file)
        input_dim = (bvh.numJoints() + 1) * vec_dim # +1 is for the root
        # input_dim = (bvh.numJoints() ) * vec_dim # +1 is for the root
        return input_dim

    def get_joint_cnt(self):
        bvh_file = self.bvh_files[0]
        bvh = BVH()
        bvh.load(bvh_file)
        return bvh.numJoints()

    def extract_root_position(self, bvh):
        root_pos = []
        for frame_idx in range(bvh.numFrames()):
            bvh.readFrame(frame_idx)
            root_pos.append([bvh.root.globalPos()])
        return np.array(root_pos)


    def compute_dataset_bbox(self):
        """
        Compute the dataset's minimum and maximum root positions
        """
        bb_min = [float('inf'), float('inf'), float('inf')]
        bb_max = [float('-inf'), float('-inf'), float('-inf')]

        for file in self.bvh_files:
            bvh = BVH()
            bvh.load(file)

            p = bvh.root.globalPos()
            # for i in range(self.numJoints()):
            #     p = self.jointById(i).globalPos()
            for j in range(3):
                if p[j] < bb_min[j]:
                    bb_min[j] = p[j]
                if p[j] > bb_max[j]:
                    bb_max[j] = p[j]
        return np.array(bb_min), np.array(bb_max)

    def extract_rotations(self, bvh):
        rotations = []
        for frame_idx in range(bvh.numFrames()):
            bvh.readFrame(frame_idx)
            frame_rotations = []
            for joint_idx in range(bvh.numJoints()):
                joint = bvh.jointById(joint_idx)
                # frame_rotations.append(joint.localRotQuat())
                frame_rotations.append(joint.localRot6D())

            rotations.append(frame_rotations)
        return np.array(rotations)

    def extract_root_and_rotations(self, bvh):
        data = []


        for frame_idx in range(bvh.numFrames()):

            bvh.readFrame(frame_idx)
            frame_data = []

            # normalized_root = (bvh.root.globalPos() - self.bb_min) / self.bb_size

            # frame_data.append(list(normalized_root) + [0])
            frame_data.append(bvh.root.globalPos() + [0]) #for quaternion


            for joint_idx in range(bvh.numJoints()):
                joint = bvh.jointById(joint_idx)


                frame_data.append(joint.localRotQuat())

            # add root to the end


            data.append(frame_data)


        return np.array(data)



    def normalize_quaternion(quat):
        norm = torch.norm(quat, dim=-1, keepdim=True)
        normalized_quat = quat / norm
        return normalized_quat

    def normalize_rotations(self, rotations):
        # Normalize each quaternion to have unit norm
        norm = np.linalg.norm(rotations, axis=-1, keepdims=True)
        normalized_rotations = rotations / norm
        return normalized_rotations


    def __len__(self):
        return len(self.bvh_files)

def triplet_collate_fn(batch):
    """
    Custom collate function to handle variable-length motion sequences.
    Args:
        batch: A list of tuples (m0, m1, m2, r_m0_m1, r_m0_m2, r_m1_m2).
               Each motion (m0, m1, m2) is a tensor representing motion data of varying lengths.
    Returns:
        Padded batch of (m0, m1, m2) with corresponding lengths and perceptual distances.
    """
    # Unpack the batch
    m0, m1, m2, r_m0_m1, r_m0_m2, r_m1_m2 = zip(*batch)

    # Convert perceptual distances to tensors
    r_m0_m1 = torch.tensor(r_m0_m1, dtype=torch.float32)
    r_m0_m2 = torch.tensor(r_m0_m2, dtype=torch.float32)
    r_m1_m2 = torch.tensor(r_m1_m2, dtype=torch.float32)

    # Get lengths of the sequences
    lengths_m0 = torch.tensor([m.shape[0] for m in m0], dtype=torch.long)
    lengths_m1 = torch.tensor([m.shape[0] for m in m1], dtype=torch.long)
    lengths_m2 = torch.tensor([m.shape[0] for m in m2], dtype=torch.long)

    # Sort by the lengths of m0 (you can choose to sort by another motion, or sort each separately)
    sorted_indices0 = lengths_m0.argsort(descending=True)
    sorted_indices1 = lengths_m1.argsort(descending=True)
    sorted_indices2 = lengths_m2.argsort(descending=True)

    # Sort the data based on the sorted lengths
    m0 = [m0[i] for i in sorted_indices0]
    m1 = [m1[i] for i in sorted_indices1]
    m2 = [m2[i] for i in sorted_indices2]
    lengths_m0 = lengths_m0[sorted_indices0]
    lengths_m1 = lengths_m1[sorted_indices1]
    lengths_m2 = lengths_m2[sorted_indices2]

    # Pad the sequences
    padded_m0 = pad_sequence(m0, batch_first=True)
    padded_m1 = pad_sequence(m1, batch_first=True)
    padded_m2 = pad_sequence(m2, batch_first=True)

    # Return the padded sequences and corresponding perceptual distances
    return padded_m0, padded_m1, padded_m2, lengths_m0, lengths_m1, lengths_m2, r_m0_m1, r_m0_m2, r_m1_m2

def triplet_collate_last_val_fn(batch):
    """
    Custom collate function for padding sequences to the same length in a batch.
    Returns padded sequences
    """
    m0, m1, m2,  r_m0_m1, r_m0_m2, r_m1_m2 = zip(*batch)



    # Convert perceptual distances to tensors
    r_m0_m1 = torch.tensor(r_m0_m1, dtype=torch.float32)
    r_m0_m2 = torch.tensor(r_m0_m2, dtype=torch.float32)
    r_m1_m2 = torch.tensor(r_m1_m2, dtype=torch.float32)

    # Get lengths of the sequences
    lengths_m0 = torch.tensor([m.shape[0] for m in m0], dtype=torch.long)
    lengths_m1 = torch.tensor([m.shape[0] for m in m1], dtype=torch.long)
    lengths_m2 = torch.tensor([m.shape[0] for m in m2], dtype=torch.long)


    max_length = max(lengths_m0.max(), lengths_m1.max(), lengths_m2.max())



    padded_m0 = pad_with_last_value(m0,  max_length)
    padded_m1 = pad_with_last_value(m1,  max_length)
    padded_m2 = pad_with_last_value(m2,  max_length)

    # Create masks for attention
    # Mask is True for padding positions and False for non-padding positions
    mask_m0 = (torch.arange(max_length).expand(len(m0), max_length) >= lengths_m0.unsqueeze(1))
    mask_m1 = (torch.arange(max_length).expand(len(m1), max_length) >= lengths_m1.unsqueeze(1))
    mask_m2 = (torch.arange(max_length).expand(len(m2), max_length) >= lengths_m2.unsqueeze(1))



    return (padded_m0, padded_m1, padded_m2,  lengths_m0, lengths_m1, lengths_m2, r_m0_m1, r_m0_m2, r_m1_m2, mask_m0, mask_m1, mask_m2)

def collate_fn(batch):

    sequences, lengths = zip(*batch)

    # lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_length = lengths.max()
    # Pad sequences to the max length in the batch
    padded_sequences = pad_with_last_value(sequences, max_length)

    mask = (torch.arange(max_length).expand(len(sequences), max_length) >= lengths.unsqueeze(1))
    return padded_sequences, lengths, mask

def pad_with_last_value(sequences, max_length):
    """
    Pad a batch of sequences with the last value of each sequence instead of zeros.
    Args:
        sequences: List of sequences (each sequence is a tensor of shape [seq_len, ...])
    Returns:
        Padded sequences: Tensor of shape [batch_size, max_seq_len, ...]
    """
    # Get the lengths of each sequence
    # lengths = torch.tensor([seq.size(0) for seq in sequences])
    # max_len = lengths.max()

    padded_sequences = []
    for seq in sequences:
        # Get the last value of the sequence
        last_value = seq[-1].unsqueeze(0)  # Shape: [1, ...]
        # Pad with the last value of the sequence
        padding_size = max_length - seq.size(0)
        if padding_size > 0:
            padding = last_value.expand(padding_size, *seq.shape[1:])  # Shape: [padding_size, ...]
            # Concatenate the original sequence with the padding
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq  # No padding needed

        padded_sequences.append(padded_seq)

    # Stack all padded sequences into a single tensor
    return torch.stack(padded_sequences, dim=0) #, lengths

def pad_with_zero(sequences, max_length):
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    # max_len = lengths.max()

    padded_sequences = []
    for seq in sequences:
        # Get the last value of the sequence
        last_value = seq[-1].unsqueeze(0)  # Shape: [1, ...]
        # Pad with the last value of the sequence
        padding_size = max_length - seq.size(0)
        if padding_size > 0:
            padding = torch.zeros(padding_size, *seq.shape[1:], dtype=seq.dtype, device=seq.device)
            # Concatenate the original sequence with the padding
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq  # No padding needed

        padded_sequences.append(padded_seq)

    # Stack all padded sequences into a single tensor
    return torch.stack(padded_sequences, dim=0)  # , lengths
