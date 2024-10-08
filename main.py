"""
Program entry point. Loads data for, trains, and evaluates both effort and similarity networks.

If command line argument is provided, program assumes value represents a parallel job index and treats the
execution as a remote machine run. Otherwise, the program is assumed to be running locally.
"""

import os
import sys
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(curr_path + '\networks')
from networks.effort_network import EffortNetwork
from networks.effort_generator import MotionDataGenerator
from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
import networks.triplet_mining as triplet_mining
import src.organize_synthetic_data as osd
import collect_job_metrics
import time
import conf

remote_sliding_window_sizes = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("running on remote machine")
        conf.num_task = sys.argv[1]
        sliding_window_sizes = remote_sliding_window_sizes
    else:
        conf.num_task = None
        sliding_window_sizes = [10]

    if conf.num_task:
        conf.all_bvh_dir = conf.REMOTE_MACHINE_DIR_VALUES['all_bvh_dir']
        conf.bvh_files_dir_walking = conf.REMOTE_MACHINE_DIR_VALUES['bv_files_dir']
        conf.effort_network_exemplars_dir = conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS['exemplars_dir'] = (conf.REMOTE_MACHINE_DIR_VALUES['exemplars_dir'] +
                                                                                                      conf.num_task + '/')
        conf.output_metrics_dir = conf.REMOTE_MACHINE_DIR_VALUES['output_metrics_dir']
        conf.checkpoint_root_dir = conf.REMOTE_MACHINE_DIR_VALUES['checkpoint_root_dir'] + conf.num_task + '/'

    if not os.path.exists(conf.output_metrics_dir):
        os.makedirs(conf.output_metrics_dir)

    for window_size in sliding_window_sizes:
        checkpoint_dir = '_'.join(filter(None, [conf.checkpoint_root_dir, str(window_size)]))
        if not os.path.exists(conf.checkpoint_root_dir):
            os.makedirs(conf.checkpoint_root_dir)
            print(f"created new effort network checkpoint directory: {checkpoint_dir}")
        conf.window_delta = window_size

        # load effort data and train effort network
        batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
        # effort_train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict,
        #                                              **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
        # effort_validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict,
        #                                                   **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
        # effort_test_generator = MotionDataGenerator(batch_ids_partition['test'], labels_dict,
        #                                             **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
        # effort_network = EffortNetwork(train_generator=effort_train_generator,
        #                                validation_generator=effort_validation_generator,
        #                                test_generator=effort_test_generator, checkpoint_dir=checkpoint_dir)
        # start_time = time.time()
        # history = effort_network.run_model_training()
        # minutes_tot_time = (time.time() - start_time) / 60
        # effort_network.write_out_training_results(minutes_tot_time)
        # collect_job_metrics.collect_job_metrics()

        for i in range(1, 5):
            print(f"ITERATION: {i}")
            if i == 1:
                bool_drop_neutral_exemplar = False
                bool_fixed_neutral_embedding = True
                squared_left_right_euc_dist = True
                squared_class_neut_euc_dist = False
            elif i == 2:
                bool_drop_neutral_exemplar = False
                bool_fixed_neutral_embedding = True
                squared_left_right_euc_dist = True
                squared_class_neut_euc_dist = True
            elif i == 3:
                bool_drop_neutral_exemplar = False
                bool_fixed_neutral_embedding = True
                squared_left_right_euc_dist = True
                squared_class_neut_euc_dist = True
            elif i == 4:
                bool_drop_neutral_exemplar = True
                bool_fixed_neutral_embedding = True
                squared_left_right_euc_dist = False
                squared_class_neut_euc_dist = False
            elif i == 5:
                bool_drop_neutral_exemplar = False
                bool_fixed_neutral_embedding = False
                squared_left_right_euc_dist = True
                squared_class_neut_euc_dist = False
            else:
                raise ValueError("Invalid iteration number")
            # load similarity data and train similarity network
            # print(f"conf.similarity_batch_size: {conf.similarity_batch_size}")
            similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar)
            triplet_mining.initialize_triplet_mining(bool_drop_neutral_exemplar, bool_fixed_neutral_embedding, squared_left_right_euc_dist, squared_class_neut_euc_dist)
            similarity_train_loader = SimilarityDataLoader(similarity_dict_partition['train'])
            similarity_validation_loader = SimilarityDataLoader(similarity_dict_partition['validation'])
            similarity_test_loader = SimilarityDataLoader(similarity_dict_partition['test'])
            similarity_network = SimilarityNetwork(train_loader=similarity_train_loader,
                                                   validation_loader=similarity_validation_loader,
                                                   test_loader=similarity_test_loader,
                                                   checkpoint_dir=checkpoint_dir)
            similarity_network.run_model_training()
            # similarity_network.evaluate()
        print(f"--------------------------------------------------------------------------")
        print(f"--------------------------------------------------------------------------")
        print(f"--------------------------------------------------------------------------")
