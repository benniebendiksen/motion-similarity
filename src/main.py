import shutil
import os
import sys

curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(curr_path + '\..\networks')
from networks.effort_network import EffortNetwork
from networks.effort_generator import MotionDataGenerator
from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
import organize_synthetic_data as osd
import collect_job_metrics
import time
import conf

remote_sliding_window_sizes = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

# effort network generator params
params = {'exemplar_dim': (100, 87),
          'batch_size': conf.batch_size_efforts_network,
          'shuffle': True,
          'exemplars_dir': conf.exemplars_dir}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        conf.num_task = sys.argv[1]
    else:
        conf.num_task = None
    print(f"task number: {conf.num_task}")
    sliding_window_sizes = [10]

    if conf.num_task:
        conf.all_bvh_dir = conf.REMOTE_MACHINE_DIR_VALUES['all_bvh_dir']
        conf.bvh_files_dir = conf.REMOTE_MACHINE_DIR_VALUES['bv_files_dir']
        conf.exemplars_dir = params['exemplars_dir'] = (conf.REMOTE_MACHINE_DIR_VALUES['exemplars_dir'] +
                                                        conf.num_task + '/')
        conf.output_metrics_dir = conf.REMOTE_MACHINE_DIR_VALUES['output_metrics_dir']
        conf.checkpoint_root_dir = conf.REMOTE_MACHINE_DIR_VALUES['checkpoint_root_dir'] + conf.num_task + '/'
        sliding_window_sizes = remote_sliding_window_sizes

    if not os.path.exists(conf.output_metrics_dir):
        os.makedirs(conf.output_metrics_dir)

    if not os.path.exists(conf.exemplars_dir):
        os.makedirs(conf.exemplars_dir)

    for window_size in sliding_window_sizes:
        checkpoint_dir = '_'.join(filter(None, [conf.checkpoint_root_dir, str(window_size)]))
        if not os.path.exists(conf.checkpoint_root_dir):
            os.makedirs(conf.checkpoint_root_dir)
            print(f"created new directory: {checkpoint_dir}")
        conf.window_delta = window_size

        # load effort data and train effort network
        batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
        print(f"number of batches: {len(labels_dict.keys())}")
        print(f"type: {type(labels_dict[1])}")
        effort_train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict, **params)
        effort_validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict, **params)
        effort_test_generator = MotionDataGenerator(batch_ids_partition['test'], labels_dict, **params)
        effort_network = EffortNetwork(train_generator=effort_train_generator, validation_generator=effort_validation_generator,
                                       test_generator=effort_test_generator, checkpoint_dir=checkpoint_dir)
        start_time = time.time()
        history = effort_network.run_model_training()
        tot_time = (time.time() - start_time) / 60
        effort_network.write_out_training_results(tot_time)
        collect_job_metrics.collect_job_metrics()

        # load similarity data and train similarity network
        similarity_dict_partition = osd.load_similarity_data()
        similarity_train_loader = SimilarityDataLoader(similarity_dict_partition['train'])
        similarity_validation_loader = SimilarityDataLoader(similarity_dict_partition['validation'])
        similarity_test_loader = SimilarityDataLoader(similarity_dict_partition['test'])
        similarity_network = SimilarityNetwork(train_loader=similarity_train_loader,
                                               validation_loader=similarity_validation_loader,
                                               checkpoint_dir=checkpoint_dir)
        similarity_network.run_model_training()
