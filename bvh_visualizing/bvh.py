# bvh.py, Aline Normoyle 2013

import glm, math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import bvh_visualizing.rotation_conversions as transforms

# Constants
M_PI_2 = math.pi * 0.5
A_EPSILON = 0.001


def clamp(x, a, b): return max(a, min(x, b))


"""
eulerAngle
Utilities for converting from euler angles to quaternion
Parameter xyz: list, or glm.vec3
   Euler angles in degrees;
   Listed in X,Y,Z order regardless of euler angle order
Returns: glm.quat
"""


def eulerAngleXYZ(xyz):
    X = glm.rotate(glm.radians(xyz[0]), glm.vec3(1, 0, 0))
    Y = glm.rotate(glm.radians(xyz[1]), glm.vec3(0, 1, 0))
    Z = glm.rotate(glm.radians(xyz[2]), glm.vec3(0, 0, 1))
    return glm.quat(X * Y * Z)


def eulerAngleXZY(xyz):
    X = glm.rotate(glm.radians(xyz[0]), glm.vec3(1, 0, 0))
    Y = glm.rotate(glm.radians(xyz[1]), glm.vec3(0, 1, 0))
    Z = glm.rotate(glm.radians(xyz[2]), glm.vec3(0, 0, 1))
    return glm.quat(X * Z * Y)


def eulerAngleYXZ(xyz):
    X = glm.rotate(glm.radians(xyz[0]), glm.vec3(1, 0, 0))
    Y = glm.rotate(glm.radians(xyz[1]), glm.vec3(0, 1, 0))
    Z = glm.rotate(glm.radians(xyz[2]), glm.vec3(0, 0, 1))
    return glm.quat(Y * X * Z)


def eulerAngleYZX(xyz):
    X = glm.rotate(glm.radians(xyz[0]), glm.vec3(1, 0, 0))
    Y = glm.rotate(glm.radians(xyz[1]), glm.vec3(0, 1, 0))
    Z = glm.rotate(glm.radians(xyz[2]), glm.vec3(0, 0, 1))
    return glm.quat(Y * Z * X)


def eulerAngleZXY(xyz):
    X = glm.rotate(glm.radians(xyz[0]), glm.vec3(1, 0, 0))
    Y = glm.rotate(glm.radians(xyz[1]), glm.vec3(0, 1, 0))
    Z = glm.rotate(glm.radians(xyz[2]), glm.vec3(0, 0, 1))
    return glm.quat(Z * X * Y)


def eulerAngleZYX(xyz):
    X = glm.rotate(glm.radians(xyz[0]), glm.vec3(1, 0, 0))
    Y = glm.rotate(glm.radians(xyz[1]), glm.vec3(0, 1, 0))
    Z = glm.rotate(glm.radians(xyz[2]), glm.vec3(0, 0, 1))
    return glm.quat(Z * Y * X)


"""
toEulerAngle
Utilities for converting from quaternion to euler angles
Parameter q: glm.quat
Returns: (rx, ry, rz) tuple representing euler angles (radians) 
"""


def toEulerAngleXYZ(q):
    M = glm.mat3(q)
    M = glm.transpose(M)  # indices are reverse -> GLM is column major
    z = 0
    y = math.asin(clamp(M[0][2], -1, 1))
    x = -math.atan2(M[1][0], M[1][1])
    if y > -M_PI_2 + A_EPSILON:
        if y < M_PI_2 - A_EPSILON:
            x = math.atan2(-M[1][2], M[2][2])
            z = math.atan2(-M[0][1], M[0][0])
        else:
            x = -x
    return x, y, z


def toEulerAngleXZY(q):
    M = glm.mat3(q)
    M = glm.transpose(M)  # indices are reverse -> GLM is column major
    x = -math.atan2(M[2][0], M[2][2])
    y = 0
    z = math.asin(clamp(-M[0][1], -1, 1))
    if z > -M_PI_2 + A_EPSILON:
        if z < M_PI_2 - A_EPSILON:
            x = math.atan2(M[2][1], M[1][1])
            y = math.atan2(M[0][2], M[0][0])
        else:
            x = -x
    return x, y, z


def toEulerAngleYXZ(q):
    M = glm.mat3(q)
    M = glm.transpose(M)  # indices are reverse -> GLM is column major
    X = math.asin(clamp(-M[1][2], -1, 1))
    Z = 0
    Y = -math.atan2(M[0][1], M[0][0])
    if X > -M_PI_2 + A_EPSILON:
        if X < M_PI_2 - A_EPSILON:
            Y = math.atan2(M[0][2], M[2][2])
            Z = math.atan2(M[1][0], M[1][1])
        else:
            Y = -Y
    return X, Y, Z


def toEulerAngleYZX(q):
    M = glm.mat3(q)
    M = glm.transpose(M)  # indices are reverse -> GLM is column major
    Z = math.asin(clamp(M[1][0], -1, 1))
    X = 0
    Y = -math.atan2(M[2][1], M[2][2])
    if Z > -M_PI_2 + A_EPSILON:
        if Z < M_PI_2 - A_EPSILON:
            Y = math.atan2(-M[2][0], M[0][0])
            X = math.atan2(-M[1][2], M[1][1])
        else:
            Y = math.atan2(M[2][1], M[2][2])
    return X, Y, Z


def toEulerAngleZXY(q):
    M = glm.mat3(q)
    M = glm.transpose(M)  # indices are reverse -> GLM is column major
    X = math.asin(clamp(M[2][1], -1, 1))
    Y = 0
    Z = -math.atan2(M[0][2], M[0][0])
    if X > -M_PI_2 + A_EPSILON:
        if X < M_PI_2 - A_EPSILON:
            Z = math.atan2(-M[0][1], M[1][1])
            Y = math.atan2(-M[2][0], M[2][2])
        else:
            Z = math.atan2(M[0][2], M[0][0])
    return X, Y, Z


def toEulerAngleZYX(q):
    M = glm.mat3(q)
    M = glm.transpose(M)  # indices are reverse -> GLM is column major
    X = math.atan2(-M[0][1], -M[0][2])
    Y = math.asin(clamp(-M[2][0], -1, 1))
    Z = 0
    if Y > -M_PI_2 + A_EPSILON:
        if Y < M_PI_2 - A_EPSILON:
            Z = math.atan2(M[1][0], M[0][0])
            X = math.atan2(M[2][1], M[2][2])
        else:
            X = math.atan2(M[0][1], M[0][2])
    return X, Y, Z


def EulerToQuat(roo, xyz):
    """
  Convert from euler angles with the given rotation order to quaternion
  roo: str
     Euler Angle rotation order, e.g. "xyz"
  xyz: list, or glm.vec3
     Euler angles (degrees)
  Returns: glm.quat
  """
    # print("roo: ", roo, "xyz: ", xyz)
    if roo == "xyz":
        return eulerAngleXYZ(xyz)
    elif roo == "xzy":
        return eulerAngleXZY(xyz)
    elif roo == "yxz":
        return eulerAngleYXZ(xyz)
    elif roo == "yzx":
        return eulerAngleYZX(xyz)
    elif roo == "zxy":
        return eulerAngleZXY(xyz)
    elif roo == "zyx":
        return eulerAngleZYX(xyz)
    else:
        print("Invalid: ", roo)

    return glm.quat(0, 0, 0, 0)


def QuatToEuler(roo, q):
    x = 0
    y = 0
    z = 0
    if roo == "xyz":
        x, y, z = toEulerAngleXYZ(q)
    elif roo == "xzy":
        x, y, z = toEulerAngleXZY(q)
    elif roo == "yxz":
        x, y, z = toEulerAngleYXZ(q)
    elif roo == "yzx":
        x, y, z = toEulerAngleYZX(q)
    elif roo == "zxy":
        x, y, z = toEulerAngleZXY(q)
    elif roo == "zyx":
        x, y, z = toEulerAngleZYX(q)
    else:
        print("Invalid: ", roo)

    x = glm.degrees(x)
    y = glm.degrees(y)
    z = glm.degrees(z)
    return [x, y, z]


def EulerTo6D(order, euler_angles):
    """
    Convert Euler angles to 6D rotation representation.

    Args:
    - euler_angles: Tensor of shape (..., 3) containing the Euler angles.
    - order: String specifying the order of rotations (default 'xyz').

    Returns:
    - A 1D array containing the 6D rotation representation.
    """
    euler = euler_angles.clone().detach()  # torch.tensor(euler_angles)
    euler = transforms.euler_angles_to_matrix(euler, order.upper())
    rot_6d = transforms.matrix_to_rotation_6d(euler)
    # # Convert Euler angles to rotation matrix
    # rot_matrices = R.from_euler(order, np.array(euler_angles)).as_matrix()
    #
    # # Convert rotation matrix to 6D (first two columns)
    # # Select first two columns of the rotation matrix and reshape to 1D
    # rot_6d = rot_matrices[..., :2].reshape(-1, 6)

    return rot_6d.tolist()


def Rot6DToEuler(order, rot_6d):
    """
        Convert 6D rotation representation to Euler angles.

        Args:
        - rot_6d: Numpy array of shape (..., 6) containing the 6D rotation representation.
        - order: String specifying the order of rotations (default 'xyz').

        Returns:
        - A tensor containing the Euler angles in radians
        """
    # Reshape 6D input to get two 3D vectors

    mat = transforms.rotation_6d_to_matrix(rot_6d)
    euler = transforms.matrix_to_euler_angles(mat, order.upper())

    return euler

    # rot_6d = rot_6d.reshape(-1, 2, 3)
    #
    # # Reconstruct the full 3x3 rotation matrix
    # # The two vectors are the first two columns of the rotation matrix
    # # We need to compute the third column as the cross product of the first two columns
    # col1 = rot_6d[:, 0]
    # col2 = rot_6d[:, 1]
    # col3 = np.cross(col1, col2)
    #
    # # Form the full rotation matrix
    # rot_matrices = np.stack([col1, col2, col3], axis=-1)
    #
    # # Convert rotation matrices to Euler angles
    # euler_angles = R.from_matrix(rot_matrices).as_euler(order)
    #
    # return euler_angles.flatten()


class Joint:
    def __init__(self):
        self.offset = []
        self.channels = []
        self.name = ""
        self.children = []
        self.parent = None
        self.id = 0
        self.channelvals = []
        self.rotOrder = "xyz"

    def setRotOrder(self, new_order):
        new_order = new_order
        self.rotOrder = new_order

        # last 3 channels
        self.channels[len(self.channels) - 3:] = [f"{axis.upper()}rotation" for axis in new_order]

    def setGlobalPos(self, pos):
        if "Xposition" in self.channels:
            self.channelvals[self.channels.index("Xposition")] = pos[0]

        if "Yposition" in self.channels:
            self.channelvals[self.channels.index("Yposition")] = pos[1]

        if "Zposition" in self.channels:
            self.channelvals[self.channels.index("Zposition")] = pos[2]

    def setLocalPos(self, pos):
        self.offset[0] = pos[0]
        self.offset[1] = pos[1]
        self.offset[2] = pos[2]

        if "Xposition" in self.channels:
            self.channelvals[self.channels.index("Xposition")] = pos[0]

        if "Yposition" in self.channels:
            self.channelvals[self.channels.index("Yposition")] = pos[1]

        if "Zposition" in self.channels:
            self.channelvals[self.channels.index("Zposition")] = pos[2]

    def localRotQuat(self):
        rot = self.localRotEuler()

        return EulerToQuat(self.rotOrder, rot)

    def localRot6D(self):
        rot = self.localRotEuler()  # always takes as xyz
        rot = torch.deg2rad(torch.tensor(rot))
        return EulerTo6D(self.rotOrder, rot)

    def setLocalRotQuat(self, q):
        # euler = QuatToEuler(self.rotOrder, q)
        mat = transforms.quaternion_to_matrix(q)
        euler = transforms.matrix_to_euler_angles(mat, self.rotOrder.upper())
        euler = torch.rad2deg(euler)

        try:
            if self.rotOrder == "xyz":
                self.channelvals[self.channels.index("Xrotation")] = euler[0]
                self.channelvals[self.channels.index("Yrotation")] = euler[1]
                self.channelvals[self.channels.index("Zrotation")] = euler[2]
            elif self.rotOrder == "xzy":
                self.channelvals[self.channels.index("Xrotation")] = euler[0]
                self.channelvals[self.channels.index("Zrotation")] = euler[1]
                self.channelvals[self.channels.index("Yrotation")] = euler[2]
            elif self.rotOrder == "zyx":
                self.channelvals[self.channels.index("Zrotation")] = euler[0]
                self.channelvals[self.channels.index("Yrotation")] = euler[1]
                self.channelvals[self.channels.index("Xrotation")] = euler[2]
            elif self.rotOrder == "zxy":
                self.channelvals[self.channels.index("Zrotation")] = euler[0]
                self.channelvals[self.channels.index("Xrotation")] = euler[1]
                self.channelvals[self.channels.index("Yrotation")] = euler[2]
            elif self.rotOrder == "yxz":
                self.channelvals[self.channels.index("Yrotation")] = euler[0]
                self.channelvals[self.channels.index("Xrotation")] = euler[1]
                self.channelvals[self.channels.index("Zrotation")] = euler[2]
            elif self.rotOrder == "yzx":
                self.channelvals[self.channels.index("Yrotation")] = euler[0]
                self.channelvals[self.channels.index("Zrotation")] = euler[1]
                self.channelvals[self.channels.index("Xrotation")] = euler[2]

        except:
            pass

    def setLocalRot6D(self, rot_6d):

        # rot = Rot6DToEuler("XYZ", rot_6d)  # we want xyz order for euler
        # rot = Rot6DToEuler("XYZ", rot_6d)  # we want xyz order for euler
        mat = transforms.rotation_6d_to_matrix(rot_6d)
        euler = transforms.matrix_to_euler_angles(mat, self.rotOrder.upper())
        rot = torch.rad2deg(euler)
        try:  # others are end sites without rotations
            if self.rotOrder == "xyz":
                self.channelvals[self.channels.index("Xrotation")] = rot[0]
                self.channelvals[self.channels.index("Yrotation")] = rot[1]
                self.channelvals[self.channels.index("Zrotation")] = rot[2]
            elif self.rotOrder == "xzy":
                self.channelvals[self.channels.index("Xrotation")] = rot[0]
                self.channelvals[self.channels.index("Zrotation")] = rot[1]
                self.channelvals[self.channels.index("Yrotation")] = rot[2]
            elif self.rotOrder == "zyx":
                self.channelvals[self.channels.index("Zrotation")] = rot[0]
                self.channelvals[self.channels.index("Yrotation")] = rot[1]
                self.channelvals[self.channels.index("Xrotation")] = rot[2]
            elif self.rotOrder == "zxy":
                self.channelvals[self.channels.index("Zrotation")] = rot[0]
                self.channelvals[self.channels.index("Xrotation")] = rot[1]
                self.channelvals[self.channels.index("Yrotation")] = rot[2]
            elif self.rotOrder == "yxz":
                self.channelvals[self.channels.index("Yrotation")] = rot[0]
                self.channelvals[self.channels.index("Xrotation")] = rot[1]
                self.channelvals[self.channels.index("Zrotation")] = rot[2]
            elif self.rotOrder == "yzx":
                self.channelvals[self.channels.index("Yrotation")] = rot[0]
                self.channelvals[self.channels.index("Zrotation")] = rot[1]
                self.channelvals[self.channels.index("Xrotation")] = rot[2]
        except:
            pass

    def localRotEuler(self):
        rot = [0, 0, 0]
        try:
            rot[0] = self.channelvals[self.channels.index("Xrotation")]
            rot[1] = self.channelvals[self.channels.index("Yrotation")]
            rot[2] = self.channelvals[self.channels.index("Zrotation")]
        except:
            pass
        return rot

    def setLocalRotEuler(self, rot):
        try:
            self.channelvals[self.channels.index("Xrotation")] = rot[0]
            self.channelvals[self.channels.index("Yrotation")] = rot[1]
            self.channelvals[self.channels.index("Zrotation")] = rot[2]
        except:
            pass

    def localPos(self):
        loc = [0, 0, 0]
        try:
            loc[0] = self.channelvals[self.channels.index("Xposition")]
            loc[1] = self.channelvals[self.channels.index("Yposition")]
            loc[2] = self.channelvals[self.channels.index("Zposition")]
            return loc
        except:
            pass
        return self.offset

    def globalPos(self):
        listp = self.localPos()
        p = glm.vec3(listp[0], listp[1], listp[2])

        parent = self.parent
        while parent != None:
            p = parent.localRotQuat() * p + parent.localPos()
            parent = parent.parent
        return [p.x, p.y, p.z]

    def globalRot(self):
        r = self.localRotQuat()

        parent = self.parent
        while parent != None:
            r = parent.localRotQuat() * r
            parent = parent.parent
        return r


class BVH:

    def __init__(self):
        self.clear()

    def clear(self):
        self.joints = []
        self.jointMap = {}
        self.root = None
        self.frames = []
        self.frameRate = 30

    def skeletonRoot(self):
        return self.root

    def load(self, filename):
        self.clear()

        file = open(filename)
        lines = file.readlines()
        if "HIERARCHY" not in lines[0]:
            return False

        parent = None
        current = None
        motion = False

        for line in lines[1:len(lines)]:
            tokens = line.split()
            if len(tokens) == 0:  # Empty line
                continue

            if tokens[0] in ["ROOT", "JOINT", "End"]:

                if current is not None:
                    parent = current

                current = Joint()
                current.name = tokens[1]

                current.id = len(self.joints)
                if current.id == 0:
                    self.root = current

                current.parent = parent
                if parent is not None:
                    current.parent.children.append(current)

                self.joints.append(current)
                self.jointMap[current.name] = current

            elif "OFFSET" in tokens[0]:
                offset = []
                for i in range(1, len(tokens)):
                    offset.append(float(tokens[i]))
                current.offset = offset

            elif "CHANNELS" in tokens[0]:
                current.channels = tokens[2:len(tokens)]
                for i in range(len(current.channels)):
                    current.channelvals.append(0)

                str = ""
                chans = list(current.channels)
                # chans.reverse() # Maya is reversed
                for channel in chans:
                    if channel == "Xrotation":
                        str += "x"
                    elif channel == "Yrotation":
                        str += "y"
                    elif channel == "Zrotation":
                        str += "z"

                current.rotOrder = str

            elif "{" in tokens[0]:
                pass

            elif "}" in tokens[0]:
                current = current.parent
                if current:
                    parent = current.parent

            elif "MOTION" in tokens[0]:
                motion = True

            elif "Frames:" in tokens[0]:
                pass

            elif "Frame" in tokens[0]:
                self.frameRate = 1.0 / float(tokens[2])

            elif motion:  # Read frame data
                vals = []
                for token in tokens:
                    vals.append(float(token))
                self.frames.append(vals)

        if self.numFrames() > 0:
            self.readFrame(0)  # IMPORTANT! Saves pose to joints

        self.findGlobalBoundaries()

    def save(self, filename):
        fileid = open(filename, "w")

        fileid.writelines("HIERARCHY\n")
        self.saveSkeleton(fileid, self.skeletonRoot())
        fileid.writelines("\n")

        fileid.writelines("MOTION\n")
        fileid.writelines("Frames: %d\n" % self.numFrames())
        fileid.writelines("Frame Time: %f\n" % (1.0 / self.frameRate))

        for each in self.frames:
            for v in each:
                fileid.writelines("%.4f " % v)
            fileid.writelines("\n")

    def saveSkeleton(self, fileid, joint, indent=""):
        # if joint.channels != None:
        if joint == self.root:
            line1 = "%sROOT %s" % (indent, joint.name)
        else:
            line1 = "%sJOINT %s" % (indent, joint.name)
        line2 = "%s{" % (indent)
        fileid.writelines(line1 + "\n")
        fileid.writelines(line2 + "\n")

        x = joint.offset[0]
        y = joint.offset[1]
        z = joint.offset[2]
        line1 = "\t%sOFFSET %.4f %.4f %.4f\n" % (indent, x, y, z)
        fileid.writelines(line1)
        if joint.channels != None and "Site" not in joint.name:
            fileid.writelines("\t%sCHANNELS %d " % (indent, len(joint.channels)))
            for channel in joint.channels:
                fileid.writelines("%s " % channel)
        fileid.writelines("\n")

        for eachChild in joint.children:
            self.saveSkeleton(fileid, eachChild, indent + "\t")
        # if joint.channels != None:
        line2 = "%s}" % (indent)
        fileid.writelines(line2 + "\n")

    def numFrames(self):
        return len(self.frames)

    def writeFrame(self, frameNum):
        """
        Write the values currently stored in each joint to the saved frames
        """
        idx = 0
        for joint in self.joints:
            for i in range(len(joint.channels)):
                v = joint.channelvals[i]
                self.frames[frameNum][idx] = v
                idx = idx + 1

    def readFrame(self, frameNum):
        """
        Loads the values for frameNum into each joint
        """
        idx = 0
        for joint in self.joints:
            for i in range(len(joint.channels)):
                v = self.frames[frameNum][idx]
                joint.channelvals[i] = v
                idx = idx + 1

    def adjustFrameRate(self, new_frame_rate):
        """
        Adjusts the frame rate of the BVH by interpolating or decimating the frames.
        If the new frame rate is higher, interpolates between frames.
        If the new frame rate is lower, decimates the frames.
        """
        if new_frame_rate == self.frameRate:
            print("New frame rate is the same as the current frame rate. No changes made.")
            return

        ratio = new_frame_rate / self.frameRate
        num_original_frames = len(self.frames)
        new_num_frames = int(num_original_frames * ratio)

        print(f"Adjusting frame rate from {self.frameRate} to {new_frame_rate}.")
        print(f"Original number of frames: {num_original_frames}, New number of frames: {new_num_frames}.")

        # Interpolation or decimation
        new_frames = []
        for i in range(new_num_frames):
            original_idx = i / ratio
            lower_idx = int(math.floor(original_idx))
            upper_idx = int(math.ceil(original_idx))

            if lower_idx == upper_idx:
                new_frames.append(self.frames[lower_idx])
            else:
                interp_factor = original_idx - lower_idx
                interpolated_frame = [(1 - interp_factor) * a + interp_factor * b for a, b in
                                      zip(self.frames[lower_idx], self.frames[upper_idx])]
                new_frames.append(interpolated_frame)

        self.frames = new_frames
        self.frameRate = new_frame_rate

    def numJoints(self):
        return len(self.joints)

    def jointById(self, idx):
        return self.joints[idx]

    def jointByName(self, jointName):
        joint = self.jointMap[jointName]
        return joint

    def findGlobalBoundaries(self):
        self.bbMin = [float('inf'), float('inf'), float('inf')]
        self.bbMax = [float('-inf'), float('-inf'), float('-inf')]

        for f in range(self.numFrames()):
            self.readFrame(f)
            # Changed this to root pos
            p = self.root.globalPos()
            # for i in range(self.numJoints()):
            #     p = self.jointById(i).globalPos()
            for j in range(3):
                if p[j] < self.bbMin[j]:
                    self.bbMin[j] = p[j]
                if p[j] > self.bbMax[j]:
                    self.bbMax[j] = p[j]

    def rotate(self, degrees, rot_vec):
        for f in range(self.numFrames()):
            self.readFrame(f)

            p = self.root.globalPos()

            rotation_matrix = glm.rotate(glm.radians(degrees), glm.vec3(rot_vec))

            rotated_pos = glm.vec3(rotation_matrix * glm.vec4(p[0], p[1], p[2], 1.0))
            self.root.setGlobalPos([rotated_pos.x, rotated_pos.y, rotated_pos.z])

            local_quat = self.root.localRotQuat()
            rotated_quat = glm.quat_cast(rotation_matrix) * local_quat
            self.root.setLocalRotQuat(torch.tensor(rotated_quat))

            # for joint in self.joints:
            #     local_quat = joint.localRotQuat()
            #     rotated_quat = glm.quat_cast(rotation_matrix) * local_quat
            #     joint.setLocalRotQuat(torch.tensor(rotated_quat))

            self.writeFrame(f)

    def change_rotation_order(self, target_order="zyx"):
        # Step 2: Iterate through each joint with rotation data
        for joint in self.joints:
            original_order = joint.rotOrder  # E.g., "ZXY"
            if original_order != target_order:
                for f in range(self.numFrames()):
                    self.readFrame(f)
                    # Extract current rotation angles

                    # angles = joint.localRotEuler()
                    # Convert to rotation matrix with original order
                    # rotation_matrix = R.from_euler(original_order.upper(), angles, degrees=True).as_matrix()

                    # Convert back to Euler angles in target order
                    # new_angles = R.from_matrix(rotation_matrix).as_euler(target_order, degrees=True)

                    # angles = torch.tensor(joint.localRotEuler())
                    # quat = joint.localRotQuat()
                    rot = joint.localRotEuler()
                    quat = EulerToQuat(joint.rotOrder, rot)

                    euler = QuatToEuler(target_order, quat)
                    # mat = transforms.euler_angles_to_matrix(torch.deg2rad(angles), original_order.upper())
                    # euler = transforms.matrix_to_euler_angles(mat, target_order.upper())
                    # euler = torch.rad2deg(euler)

                    # Update the frame data
                    joint.setLocalRotEuler(euler)
                    self.writeFrame(f)

        for joint in self.joints:
            joint.setRotOrder(target_order)
