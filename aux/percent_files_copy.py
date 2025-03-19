import os
import conf_2
import random
import shutil

# Copy a percentage of input directory files to output directory
PERCENT_FILES_TO_COPY = 12.5
FILE_COUNT = 0
NUM_SAMPLES = None

def rename_blender_to_unity_files(dir):
    for f in os.listdir(dir):
        if "|" in f:
            sub_strings = f.split("|")
            os.rename(dir + f, dir + sub_strings[1])

def run_percent_files_copy(task_num):
    # stylized jumping + polarized stylized files: count: 14838
    # num samps at % 6.25: 911
    # after window application num_samps: data shape: (81200, 100, 87)
    # num samps at % 12.5: 1830
    # after window application num_samps: data shape: (160454, 100, 87)
    percent_copy = 6.0 + int(task_num)*2
    conf.percent_files_copied = percent_copy
    INPUT_DIR = conf.all_bvh_dir
    OUTPUT_DIR = conf.bvh_files_dir_walking + task_num
    file_count = 0
    for _, _ in enumerate(os.listdir(INPUT_DIR)):
        file_count += 1
    # print(f"count: {file_count}")
    NUM_SAMPLES = int(percent_copy / 100 * file_count)
    # print(f"num samps: {NUM_SAMPLES}")

    if not os.path.exists(conf.bvh_files_dir_walking):
        os.mkdir(conf.bvh_files_dir_walking)
        os.mkdir(OUTPUT_DIR)
        print(f"created new directories: {conf.bvh_files_dir_walking} , {OUTPUT_DIR}")
    elif not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print(f"created new directory: {OUTPUT_DIR}")

    filenames = os.listdir(INPUT_DIR)

    # Manually specify polarized states and drives as they are needed for training the similarity network
    files_to_copy = ['file1.txt', 'file2.txt', 'file3.txt']

    # Select random files from the remaining files
    remaining_files = list(set(filenames) - set(files_to_copy))
    random_files = random.sample(remaining_files, NUM_SAMPLES - len(files_to_copy))

    # Combine the specific files and random files to create the final list
    file_subset_names = files_to_copy + random_files

    for file_name in file_subset_names:
        print(f"copying: {file_name}")
        shutil.copy(os.path.join(INPUT_DIR, file_name), os.path.join(OUTPUT_DIR))




