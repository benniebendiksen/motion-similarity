
from bvh_visualizing.MotionDataset import MotionDataset
from bvh_visualizing.bvh import BVH
from bvh_visualizing.conf import vec_dim
import torch
import os


class BVHDataset(MotionDataset):
    def __init__(self, directory):
        # super().__init__(csv_file, data_folder)
        self.directory = directory

        self.bvh_files = [os.path.join( root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(".bvh")]

        self.input_dim = self.get_input_dim(vec_dim=vec_dim) # 4 is for quaternion

        self.joint_cnt = self.get_joint_cnt()

        #for normalization

        # self.bb_min, self.bb_max = self.compute_dataset_bbox()
        # self.bb_size = self.bb_max - self.bb_min

        print(len(self.bvh_files))
        # self.load_into_memory()

    def __getitem__(self, idx):


        # positions = self.extract_global_positions(animation)
        # root_positions = self.extract_root_position(animation)


        animation = BVH()
        animation.load(self.bvh_files[idx])
        # rotations = self.extract_rotations(animation)

        motion_data= self.extract_root_and_rotations(animation)

        # limb_indices = self.get_limb_indices(animation)
        # normalized_rotations = self.normalize_rotations(rotations)

        # del animation

        # Run garbage collection to free up memory
        # gc.collect()

        # return torch.tensor(self.features[idx], dtype=torch.float32), self.features[idx].shape[0]
        return torch.tensor(motion_data, dtype=torch.float32), motion_data.shape[0]


