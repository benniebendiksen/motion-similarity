import os
import conf_2


def rename_blender_to_unity_files(dir):
    for f in os.listdir(dir):
        if "|" in f:
            sub_strings = f.split("|")
            os.rename(conf.bvh_files_dir_walking + f, conf.bvh_files_dir_walking + sub_strings[1])

if __name__ == "__main__":
    rename_blender_to_unity_files(conf.bvh_files_dir_walking)