import shutil
from datetime import datetime
import keras
from keras.layers import Flatten, Dense, Conv1D, MaxPooling1D, BatchNormalization, Dropout
from keras.optimizers import Adam
from networks.effort_network import EffortNetwork
from keras import models, Sequential
from networks.generator import MotionDataGenerator
import organize_synthetic_data as osd
import percent_files_copy
import time
import conf
import sys
import os


def delete_exemplars_dir(task_num):
    directory = conf.exemplars_dir + task_num + '/'
    if os.path.exists(directory):
        if not os.listdir(directory):
            print(f"Directory is empty: {directory}")
        else:
            shutil.rmtree(directory)
            print(f"Deleted directory: {directory}")
    else:
        print(f"Directory does not exist: {directory}")

# function used for repeating specific remote compute job instances
def temp_test_check_run_repeatability(task_num):
    test_int = int(task_num)
    if test_int == 1:
        test_int = 3
        sliding_size = [8]
    elif test_int == 2:
        test_int = 9
        sliding_size = [8]
    elif test_int == 3:
        test_int = 12
        sliding_size = [10]
    elif test_int == 4:
        test_int = 13
        sliding_size = [6]
    return test_int, sliding_size


# sliding_window_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

# generator params
params = {'exemplar_dim': (100, 91),
          'batch_size': conf.batch_size,
          'shuffle': True}

if __name__ == '__main__':
    #pass in index of current task
    conf.task_num = sys.argv[1]
    #indicate if running remotely or not
    conf.remote = sys.argv[2]
    test_int, sliding_window_sizes = temp_test_check_run_repeatability(conf.task_num)
    conf.task_num = str(test_int)
    params['exemplars_dir'] = conf.exemplars_dir + conf.task_num + '/'
    checkpoint_dir = conf.checkpoint_dir + conf.task_num
    percent_files_copy.run_percent_files_copy(conf.task_num)
    for window_size in sliding_window_sizes:
        delete_exemplars_dir(conf.task_num)
        conf.window_delta = window_size
        batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False, task_num=conf.task_num)
        print(f"number of batches: {len(labels_dict.keys())}")
        print(f"type: {type(labels_dict[1])}")
        train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict, **params)
        validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict, **params)
        test_generator = MotionDataGenerator(batch_ids_partition['test'], labels_dict, **params)
        effort_network = EffortNetwork(two_d_conv=False, model_num=1)
        start_time = time.time()
        history = effort_network.run_model_training(train_generator, validation_generator,
                                                    checkpoint_dir)
        tot_time = (time.time() - start_time) / 60
        index_window_size = conf.task_num + '.' + str(window_size)
        effort_network.write_out_eval_accuracy(test_generator, conf.task_num, checkpoint_dir, tot_time)
        # saved_model = effort_network.models.load_model(checkpoint_dir)
        # saved_model.load_weights(checkpoint_dir)
        # test_loss, test_acc = saved_model.evaluate(validation_generator)
        # print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
