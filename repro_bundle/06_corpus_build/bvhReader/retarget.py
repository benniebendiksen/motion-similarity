import os
import glm
import copy
from bvh import *
from bvhReader.bvhvisualize import *
import numpy as np
import glm
import torch

def apply_scaled_root_motion(source_bvh, target_bvh, joint_mapping):
    """
    Correctly transfers and scales root motion from source to target.
    This avoids cumulative errors and uses a stable, T-pose based scale factor.

    Args:
        source_bvh (BVH): The source animation BVH.
        target_bvh (BVH): The target skeleton BVH (after rotations have been retargeted).
        joint_mapping (dict): The mapping of source to target joint names.
    """
    print("Applying stable scaled root motion...")

    # 1. Calculate a SINGLE, STABLE scale factor based on T-pose hip height.
    # This represents the overall size difference between the characters.
    source_bvh.readFrame(0)
    target_bvh.readFrame(0)

    # Mapping is {source_name: target_name}; find the source joint whose
    # target is the root of the target skeleton.
    target_root_name = target_bvh.root.name
    source_body_root_name = next(
        (src for src, tgt in joint_mapping.items() if tgt == target_root_name), None
    )
    if not source_body_root_name:
        raise ValueError(f"Mapping missing for target root: '{target_root_name}'")



    try:
        # Define standard names for leg joints to look for in the target skeleton
        target_hip_name = 'RightUpLeg'
        target_foot_name = 'RightFoot'

        source_hip_name = None
        source_foot_name = None

        # Find the corresponding source joint names from the mapping
        for s_name, t_name in joint_mapping.items():
            if t_name == target_hip_name:
                source_hip_name = s_name
            if t_name == target_foot_name:
                source_foot_name = s_name

        if not source_hip_name or not source_foot_name:
            # Raise an error to be caught, triggering the fallback.
            raise NameError(f"Required leg joints ('{target_hip_name}', '{target_foot_name}') not found in mapping.")

        # Calculate leg lengths from T-pose
        source_hip_pos = np.array(source_bvh.jointByName(source_hip_name).globalPos())
        source_foot_pos = np.array(source_bvh.jointByName(source_foot_name).globalPos())
        source_leg_length = np.linalg.norm(source_hip_pos - source_foot_pos)

        target_hip_pos = np.array(target_bvh.jointByName(target_hip_name).globalPos())
        target_foot_pos = np.array(target_bvh.jointByName(target_foot_name).globalPos())
        target_leg_length = np.linalg.norm(target_hip_pos - target_foot_pos)

        print(f"Source leg length: {source_leg_length:.2f}, Target leg length: {target_leg_length:.2f}")

        if source_leg_length > 1e-5:  # Use a small epsilon to avoid division by zero
            scale_factor = target_leg_length / source_leg_length
        else:
            raise ValueError("Source leg length is effectively zero.")

    except (NameError, ValueError, KeyError) as e:
        print(f"Warning: Could not use leg length for scaling ({e}).")
        print("Falling back to root height comparison.")

        # Fallback to the original method
        source_height = source_bvh.jointByName(source_body_root_name).globalPos()[1]
        target_height = target_bvh.jointByName(target_bvh.root.name).globalPos()[1]

        if source_height != 0:
            scale_factor = target_height / source_height
        else:
            print("Warning: Source T-pose height is 0. Using scale of 1.0.")
            scale_factor = 1.0


    print(f"Calculated stable scale factor: {scale_factor:.4f}")

    # 2. Get the initial T-pose positions.
    source_start_pos = np.array(source_bvh.jointByName(source_body_root_name).globalPos())
    target_start_pos = np.array(target_bvh.jointByName(target_bvh.root.name).globalPos())

    # 3. Iterate through frames and apply the scaled motion delta.
    for f in range(source_bvh.numFrames()):
        source_bvh.readFrame(f)
        target_bvh.readFrame(f)  # Read to get access to the target's root joint object

        # Get the source's current position
        source_current_pos = np.array(source_bvh.jointByName(source_body_root_name).globalPos())

        # Calculate the total travel of the source from its starting point
        source_travel = source_current_pos - source_start_pos

        # Scale this travel vector and add it to the target's starting position
        new_target_pos = target_start_pos + (source_travel * scale_factor)

        # Set this as the new position for the target's root
        # This is a direct, non-cumulative update for each frame.
        target_bvh.root.setGlobalPos(list(new_target_pos))  # For the root, localPos is globalPos

        target_bvh.writeFrame(f)

    print("Root motion applied successfully.")
    return target_bvh

def find_rotation_from_to_limb(jointFrom, jointTo):
    # Root is handled differently
    if jointFrom.parent is None or jointTo.parent is None:
        return

    limbFrom = np.array(jointFrom.parent.globalPos()) - np.array(jointFrom.globalPos())
    limbTo = np.array(jointTo.parent.globalPos()) - np.array(jointTo.globalPos())

    # Get orientations of joints in the world space
    v1 = np.array(limbFrom, dtype=np.float64)
    v2 = np.array(limbTo, dtype=np.float64)

    # Normalize the vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return glm.quat(1, 0, 0, 0)  # Identity quaternion (no rotation)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Dot product to check alignment
    dot = np.dot(v1, v2)

    if np.isclose(dot, 1.0, atol=1e-6):  # Aligned
        return glm.quat(1, 0, 0, 0)  # Identity quaternion
    elif np.isclose(dot, -1.0, atol=1e-6):  # Anti-aligned
        # Rotate 180° around any orthogonal axis
        orthogonal_axis = np.array([1, 0, 0]) if abs(v1[0]) < 0.99 else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal_axis)
        axis = axis / np.linalg.norm(axis)
        return glm.angleAxis(math.pi, glm.vec3(*axis))

    # Compute the cross product and angle
    cross = np.cross(v1, v2)
    axis = glm.vec3(*cross / np.linalg.norm(cross))

    angle = math.acos(np.clip(dot, -1.0, 1.0))

    quat_rot = glm.angleAxis(angle, axis)

    quat_rot = glm.normalize(quat_rot)
    return quat_rot

def find_rotation_tpose(jointFrom, jointTo):
    """Rest-pose alignment from skeleton OFFSETS only (true T-pose bone
    vectors), not frame-0 global positions.  In the T-pose (all rotations
    zero) the bone vector pointing from a joint back to its parent equals
    -joint.offset, so the two skeletons' structural diff is recoverable
    from the offsets alone — free of contamination from the source
    animation's starting pose.
    """
    if jointFrom.parent is None or jointTo.parent is None:
        return glm.quat(1, 0, 0, 0)

    v1 = -np.array(jointFrom.offset, dtype=np.float64)
    v2 = -np.array(jointTo.offset, dtype=np.float64)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return glm.quat(1, 0, 0, 0)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = float(np.dot(v1, v2))

    if np.isclose(dot, 1.0, atol=1e-6):
        return glm.quat(1, 0, 0, 0)
    if np.isclose(dot, -1.0, atol=1e-6):
        orthogonal = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1, orthogonal)
        axis = axis / np.linalg.norm(axis)
        return glm.angleAxis(math.pi, glm.vec3(*axis))

    cross = np.cross(v1, v2)
    axis = glm.vec3(*(cross / np.linalg.norm(cross)))
    angle = math.acos(np.clip(dot, -1.0, 1.0))
    return glm.normalize(glm.angleAxis(angle, axis))


def align_limbs(jointFromName, bvhFrom, bvhTo, joint_mapping):
        # Always fetch the source joint so we can recurse through its children.
        if jointFromName not in bvhFrom.jointMap:
            return
        jointFrom = bvhFrom.jointByName(jointFromName)

        # Only transfer rotation if this source joint has a mapping entry
        # AND the mapped target joint actually exists in the target skeleton.
        if jointFromName in joint_mapping:
            target_name = joint_mapping[jointFromName]
            if target_name in bvhTo.jointMap:
                jointTo = bvhTo.jointByName(target_name)
                if jointTo is not bvhTo.root:
                    # Structural-only rest_delta from skeleton offsets — NOT
                    # from frame 0 of the source animation.  Using frame 0
                    # would conflate structural differences with the source's
                    # starting pose, which e.g. leaves KIT arms stuck at
                    # T-pose (wide) because their frame-0 "arms-down" pose
                    # gets absorbed into rest_delta and cancels out.
                    rest_delta = find_rotation_tpose(jointTo, jointFrom)

                    for f in range(0, bvhFrom.numFrames()):
                        bvhFrom.readFrame(f)
                        bvhTo.readFrame(f)
                        global_delta = find_rotation_from_to_limb(jointTo, jointFrom)
                        corrected_delta = global_delta * glm.inverse(rest_delta)
                        jointTo.parent.setGlobalRot(corrected_delta * jointTo.parent.globalRot())
                        bvhTo.writeFrame(f)

        # Always recurse, even for unmapped joints, so deeper joints are reached.
        for childFrom in jointFrom.children:
            if "Site" not in childFrom.name:
                align_limbs(childFrom.name, bvhFrom, bvhTo, joint_mapping)



# --- Main Retargeting Function ---
def retarget_bvh(source_bvh, target_bvh, joint_mapping, output_filename):
    """
    Retargets motion from a source BVH to a target BVH with a different skeleton.

    Args:
        source_bvh (BVH): The loaded BVH object with the source animation.
        target_bvh (BVH): The loaded BVH object of the target skeleton (in a T-pose).
        joint_mapping (dict): Maps source joint names to target joint names.
        output_filename (str): The path to save the resulting BVH file.
    """
    print("Starting retargeting process...")

    # 1. Prepare Target BVH Frames
    # ---------------------------------
    # Duplicate the target's T-pose for the entire duration of the source animation.
    num_frames = source_bvh.numFrames()
    t_pose_frame_data = copy.deepcopy(target_bvh.frames[0])
    target_bvh.frames = [copy.deepcopy(t_pose_frame_data) for _ in range(num_frames)]
    print(f"Target BVH prepared with {num_frames} frames.")
    # Get the global Y position (height) of the root joints from their original files
    # Mapping is {source_name: target_name}; find the source joint whose
    # target is the root of the target skeleton.
    target_root_name = target_bvh.root.name
    source_body_root_name = next(
        (src for src, tgt in joint_mapping.items() if tgt == target_root_name), None
    )
    if source_body_root_name is None:
        raise ValueError(f"No source joint maps to target root '{target_root_name}'")

    for f in range(0, source_bvh.numFrames()):
        source_bvh.readFrame(f)
        source_root = source_bvh.jointByName(source_body_root_name)
        target_bvh.root.setGlobalPos(source_root.globalPos())
        target_bvh.root.setGlobalRot(source_root.globalRot())
        target_bvh.writeFrame(f)

    # 2. Calculate T-Pose Offsets
    # ---------------------------------
    print("Calculating T-pose rotational offsets...")

    align_limbs(source_body_root_name, source_bvh, target_bvh, joint_mapping)

    # should be called after the retargeting -- to make sure the translations align
    apply_scaled_root_motion(source_bvh,target_bvh, joint_mapping)


    # 5. Save the Result
    # ---------------------------------
    print(f"Saving retargeted animation to {output_filename}...")
    target_bvh.save(output_filename)
    print("Retargeting complete! ✨")

def process_file(source_file, target_file, output_file, joint_mapping):
    source_motion = BVH()
    # source_motion.load("../datasets/bandai/dataset-1_bow_active_001.bvh")
    source_motion.load(source_file)

    target_skeleton = BVH()
    target_skeleton.load(target_file)



    retarget_bvh(source_motion, target_skeleton, joint_mapping, output_file)
    # bvh = BVH()
    # bvh.load("retargeted_animation.bvh")
    #
    # # anim = BVHAnimator(source_motion)
    # anim = BVHAnimator(bvh)
if __name__ == '__main__':
    joint_mapping = {
            'Hips': 'Hips', # bandai's root is supposed to be hips as well. joint_root is static at 0,0,0 - don't use
            'Spine': 'LowerBack',
            'Chest': 'Spine1',
            'Neck': 'Neck1',
            'Head': 'Head',
            'Shoulder_R': 'RightShoulder',
            'UpperArm_R': 'RightArm',
            'LowerArm_R': 'RightForeArm',
            'Hand_R': 'RightHand',
            'Shoulder_L': 'LeftShoulder',
            'UpperArm_L': 'LeftArm',
            'LowerArm_L': 'LeftForeArm',
            'Hand_L': 'LeftHand',
            'UpperLeg_R': 'RightUpLeg',
            'LowerLeg_R': 'RightLeg',
            'Foot_R': 'RightFoot',
            'Toes_R': 'RightToeBase',
            'UpperLeg_L': 'LeftUpLeg',
            'LowerLeg_L': 'LeftLeg',
            'Foot_L': 'LeftFoot',
            'Toes_L': 'LeftToeBase',
        }

    target_skeleton = BVH()
    target_skeleton.load('../datasets/cmuTPose.bvh')

    directory = "../datasets/bandai"
    out_directory = "../datasets/bandai_organized"

    bvh_files = [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if f.endswith(".bvh")]

    for f in bvh_files:
        source_motion = BVH()
        source_motion.load(f)

        output_file = os.path.join(out_directory, os.path.basename(f))

        if not os.path.exists(output_file):
            retarget_bvh(source_motion, target_skeleton, joint_mapping, output_file)
