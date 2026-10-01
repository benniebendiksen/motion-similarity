from bvh import *
import os
from bvhReader.bvhvisualize import *
class BVHConverterToPerformFormat(BVH):
    def __init__(self):
        super(BVHConverterToPerformFormat, self).__init__()

    def convert_to_target_format(self):
        """
        Convert the current BVH file's skeleton to match the target format (previous format).
        keep_names: if we want to keep the names as in Perform format. This isn't necessary.
        """
        # These will be dropped from cmu
        self.drop_joint_frames("Neck")
        self.drop_joint_frames("RThumb")
        self.drop_joint_frames("LThumb")

        # These will be dropped from perform
        # self.drop_joint_frames("LeftHandIndex2")
        # self.drop_joint_frames("RightHandIndex2")



    def drop_joint_frames(self, joint_name):
        """
        Drops frame values for a specific joint and removes its channels from the motion data.
        """
        # Locate the joint
        joint = self.jointByName(joint_name)
        if not joint:
            raise ValueError(f"Joint '{joint_name}' not found in the skeleton.")

        # Calculate the indices of the joint's channels in the motion data
        channel_start_index = 0
        for j in self.joints:
            if j == joint:
                break
            if j.channels:
                channel_start_index += len(j.channels)
        channel_end_index = channel_start_index + len(joint.channels)

        # Remove channels from each frame
        for i in range(len(self.frames)):
            self.frames[i] = self.frames[i][:channel_start_index] + self.frames[i][channel_end_index:]



        # Update the joint to have no channels
        self.jointByName(joint_name).channels = None
        self.jointByName(joint_name).offset = None


        # We have to keep the order of children as they correspond to frames in order
        parent_joint = joint.parent
        if parent_joint:
            # Find the index of the joint in the parent's children list
            joint_index = parent_joint.children.index(joint)

            # Insert the joint's children at the same position
            for child in reversed(joint.children):
                if child.name != "Site":
                    parent_joint.children.insert(joint_index, child)

            # Remove the joint from the parent's children list
            parent_joint.children.remove(joint)


        self.joints.remove(joint)
#


def modify_end_sites(bvh_data):
    """
    Modify a BVH hierarchy to rename all "JOINT End" to "End Site" and remove channels from End Site nodes.

    Args:
        bvh_data (str): The BVH file content as a string.

    Returns:
        str: The modified BVH content.
    """
    lines = bvh_data.splitlines()
    modified_lines = []
    inside_end_joint = False  # Track if we're inside an End node

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("JOINT Site"):
            # Rename "JOINT End" to "End Site"
            modified_lines.append(line.replace("JOINT Site", "End Site", 1))
            inside_end_joint = True  # Start tracking this node
        elif stripped.startswith("JOINT End Site"):
            # Rename "JOINT End" to "End Site"
            modified_lines.append(line.replace("JOINT End Site", "End Site", 1))
            inside_end_joint = True  # Start tracking this node
        elif stripped.startswith("Site"):
            # Rename "JOINT End" to "End Site"
            modified_lines.append(line.replace("Site", "End Site", 1))
            inside_end_joint = True  # Start tracking this node
        elif inside_end_joint and stripped.startswith("CHANNELS"):
            # Remove the channels for End Sites
            continue  # Skip this line
        elif stripped == "}":
            inside_end_joint = False  # Exit End Site
            modified_lines.append(line)
        else:
            modified_lines.append(line)

    return "\n".join(modified_lines)



def prepare_files(directory, out_directory):
    os.makedirs(out_directory, exist_ok=True)
    bvh_files = [os.path.join( root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(".bvh")]
    bvh_converter = BVHConverterToPerformFormat()
    #
    for f in bvh_files:
        bvh_converter.load(f)  # Load the source BVH file

        bvh_converter.convert_to_target_format()  # Convert the skeleton format
        bvh_converter.adjustFrameRate(30)
        out_file = os.path.join(out_directory, os.path.basename(f))

        bvh_converter.save(out_file)  # Save the converted BVH
        # print(bvh_converter.numJoints())

    # fix format issues

def fix_end_sites(dir):
    bvh_files = [os.path.join( root, f) for root, _, files in os.walk(dir) for f in files if f.endswith(".bvh")]
    for f in bvh_files:
        # Load BVH file content
        fix_end_sites_in_file(f, dir)



def fix_end_sites_in_file(f, dir):
    with open(f, "r") as file:
        bvh_content = file.read()

        # Modify the hierarchy
        modified_content = modify_end_sites(bvh_content)

        # Save to the output directory
        output_path = os.path.join(dir, os.path.basename(f))
        with open(output_path, "w") as file:
            file.write(modified_content)
        print(f"Processed and saved: {output_path}")

def reorder_bvh_hierarchy(bvh_obj, new_hierarchy):
    """
    Reorders the skeleton hierarchy in the BVH object based on a new joint order
    and adjusts the motion frames accordingly, including reordering children.

    Args:
        bvh_obj (BVH): The BVH object containing the parsed data.
        new_hierarchy (list): The list of joint names in the desired order.

    Returns:
        BVH: The updated BVH object with reordered hierarchy and frames.
    """
    # Map joints by name for easy access
    joint_map = {joint.name: joint for joint in bvh_obj.joints}


    # Reorder frames according to the hierarchy
    # Reorder motion frames
    frame_map = {}
    channel_offset = 0

    # current frame mapping
    for joint in bvh_obj.joints:
        if joint.channels:
            num_channels = len(joint.channels)
            frame_map[joint.name] = slice(channel_offset, channel_offset + num_channels)
            channel_offset += num_channels


    reordered_frames = []
    for frame in bvh_obj.frames:
        reordered_frame = []
        for joint_name in new_hierarchy:
            if joint_name in frame_map:
                reordered_frame.extend(frame[frame_map[joint_name]])
        reordered_frames.append(reordered_frame)

    # Update the motion frames in the BVH object
    bvh_obj.frames = reordered_frames


    #Reorder the skeleton to write in the file
    reordered_joints = [joint_map[joint_name] for joint_name in new_hierarchy]

    # Update parent-child relationships
    for joint in reordered_joints:
        if joint.parent:
            parent_name = joint.parent.name
            if parent_name in new_hierarchy:
                joint.parent = joint_map[parent_name]
            else:
                joint.parent = None
        # Sort children based on the new hierarchy order
        joint.children = sorted(
            [child for child in joint.children if child.name in new_hierarchy],
            key=lambda x: new_hierarchy.index(x.name)
        )

    # Update the joint list in the BVH object
    bvh_obj.joints = reordered_joints

    return bvh_obj


def fix_hierarchy(directory, out_directory):
    bvh_files = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(".bvh")]
    os.makedirs(out_directory, exist_ok=True)
    # If we want to reorder cmu
    # new_joint_order = [
    #     "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "Site",
    #     "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase","Site",
    #     "LowerBack", "Spine", "Spine1",
    #     "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand","LeftFingerBase", "LeftHandIndex1","Site",
    #     "Neck1", "Head", "Site",
    #     "RightShoulder", "RightArm", "RightForeArm", "RightHand","RightFingerBase", "RightHandIndex1","Site"
    #
    # ]


    for bvh_file in bvh_files:
        output_file = os.path.join(out_directory, os.path.basename(bvh_file))
        fix_hierarchy_perform_file(bvh_file, output_file)
        print(f"Processed: {output_file}")

def fix_hierarchy_perform_file(bvh_file, output_file):
    new_joint_order = [
        "Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "Site",
        "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site",
        "Spine", "Spine1", "Spine2",
        "Neck", "Head", "Site",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandIndex1", "LeftHandIndex3", "Site",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandIndex1", "RightHandIndex3", "Site"

    ]
    bvh = BVH()

    bvh.load(bvh_file)  # Load the BVH file into the BVH object

    reordered_bvh = reorder_bvh_hierarchy(bvh, new_joint_order)  # Reorder the hierarchy

    reordered_bvh.save(output_file)  # Save the reordered BVH to the output directory

    print(f"Processed: {output_file}")


def add_extra_joints(directory, out_directory, new_joint_names, new_joint_childs, offsets, rots):
    bvh_files = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(".bvh")]
    os.makedirs(out_directory, exist_ok=True)

    bvh = BVH()
    for bvh_file in bvh_files:
        bvh.load(bvh_file)  # Load the BVH file into the BVH object
        output_file = os.path.join(out_directory, os.path.basename(bvh_file))

        for i in range(len(new_joint_names)):
            bvh.insert_joint(new_joint_names[i], new_joint_childs[i], offsets, rots)
        bvh.save(output_file)
        print(f"Processed: {output_file}")


def retarget_motions(directory, out_directory, cmu_file):
    bvh_files = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(".bvh")]
    os.makedirs(out_directory, exist_ok=True)

    bvhFrom = BVH()
    for bvh_file in bvh_files:
        bvhCmu = BVH()
        bvhCmu.load(cmu_file) # each time create a new cmu file
        bvhFrom.load(bvh_file)  # Load the BVH file into the BVH object
        bvhFrom.find_alignment_rotations(bvhCmu)
        output_file = os.path.join(out_directory, os.path.basename(bvh_file))
        bvhCmu.save(output_file)
        print(f"Processed: {output_file}")

def check_nans(directory):

    bvh_files = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(".bvh")]

    bvh = BVH()
    for bvh_file in bvh_files:
        bvh.load(bvh_file)
        if torch.any(torch.isnan(torch.tensor(bvh.frames))):
            # print((bvh.frames))
            print(bvh_file)


###########################################
# Drop extraneous joints
###########################################
# directory = "../datasets/cmu_all"
# out_directory = "../datasets/cmu_all_perform_2"
#
# directory = "../datasets/cmu_small"
# out_directory = "../datasets/cmu_small_perform"
# #
# # # directory = "../datasets/cmu_small_walk"
# # # out_directory = "../datasets/cmu_small_walk_perform"
# #
# #
# # # directory = "../datasets/cmu_small_walk"
# # # out_directory = "../datasets/cmu_small_walk_perform"
# #
# prepare_files(directory, out_directory)
# fix_end_sites(out_directory)


#######################################
# Fix hierarchy order for perform only
###########################################

# out_directory = "../datasets/lma_perform_reorganized"
# fix_hierarchy(out_directory, out_directory )
# fix_end_sites(out_directory)

# fix_hierarchy_perform_file("../datasets/test/Tpose.bvh", "../datasets/test/Tposefixed.bvh")

################## Working- Change rotation order in perform##########################
# bvh = BVH()
# bvh.load("../datasets/test/Walking_0_0_0_0.bvh")
# bvh.change_rotation_order("zyx")
# bvh.save("../datasets/test/perform_updated.bvh")
# bvh.load("../datasets/test/perform_updated.bvh")
# anim = BVHAnimator(bvh)
######################################################

#######
# Drop extra joints from perform -index2
# bvh_converter = BVHConverterToPerformFormat()
# bvh_converter.load("../datasets/test/Tpose_0_0_0_0.bvh")  # Load the source BVH file
# bvh_converter.convert_to_target_format()  # Convert the skeleton format
# bvh_converter.adjustFrameRate(30)
# bvh_converter.save("../datasets/test/Tpose.bvh")  # Save the converted BVH


#######################################
# Add extra hip joints to perform
###########################################
# directory = "../datasets/lma_perform_reorganized"
# out_directory = "../datasets/lma_perform_reorganized_2"
# add_extra_joints(directory, out_directory, ["LHipJoint","RHipJoint"] ,["LeftUpLeg","RightUpLeg"], [0,0,0], ["Zrotation", "Xrotation", "Yrotation"])


# #######################Format perform motions with cmu skeletons ################################
# directory = "../datasets/lma_perform_reorganized_2"
# out_directory = "../datasets/lma_perform_reorganized_cmu"
#
# bvhCmu = BVH()
# bvhCmu.load("../datasets/cmuTPose.bvh")
# retarget_motions(directory, out_directory, "../datasets/cmuTPose.bvh")

######################## View animations
# #
# bvh = BVH()
# bvh.load("../datasets/lma_perform_reorganized_cmu/PointingMirror_-1_-1_1_-1.bvh")
# anim = BVHAnimator(bvh)

###############
#Print nans
# check_nans("../datasets/lma_perform_reorganized_cmu")
###############################

#
# bvhTo = BVH()
# bvhTo.load("../datasets/cmuTPose.bvh")
# bvhFrom = BVH()
# bvhFrom.load("../datasets/lma_perform_reorganized_cmu/Picking_1_-1_1_1.bvh")
anim = BVHAnimator(bvhFrom)
# bvhFrom.load("../datasets/lma_perform_reorganized_2/Pointing_0_-1_-1_1.bvh")
#
# bvhFrom.find_alignment_rotations(bvhTo)
#

# anim = BVHAnimator(bvhTo)




# anim.ani.save("../gifs/tpos.gif", writer='Pillow', fps=30)