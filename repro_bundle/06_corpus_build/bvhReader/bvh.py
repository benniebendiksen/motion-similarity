# bvh.py, Aline Normoyle 2013

import glm, math
import numpy as np
import torch
import copy
from scipy.spatial.transform import Rotation as R

import bvhReader.rotation_conversions as transforms

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

    q = EulerToQuat(order, euler_angles)  # glm.quat

    # Convert quaternion to a 3×3 rotation matrix
    # (glm.mat3_cast returns a glm.mat3 from a glm.quat)
    R = glm.mat3_cast(q)  # R is 3×3, row‐major

    col0 = glm.vec3(R[0][0], R[1][0], R[2][0])
    col1 = glm.vec3(R[0][1], R[1][1], R[2][1])

    # Return as a flat Python list [r00, r10, r20,  r01, r11, r21]
    return [col0.x, col0.y, col0.z,
            col1.x, col1.y, col1.z]


def Rot6DToEuler(order, rot6d):
    """"
    Convert 6D rotation representation to Euler angles.

    Args:
      - order: String specifying rotation order, e.g. 'xyz'
      - rot6d: Iterable of length 6 [r00, r10, r20,  r01, r11, r21]

    Returns:
      - List of 3 Euler angles [α, β, γ] in degrees, following `order`.
    """
    # # unpack the two columns
    # # these were: col0 = (R[0,0], R[1,0], R[2,0])
    # #             col1 = (R[0,1], R[1,1], R[2,1])
    # c0 = glm.vec3(rot6d[0], rot6d[1], rot6d[2])
    # c1 = glm.vec3(rot6d[3], rot6d[4], rot6d[5])
    #
    # # re-orthonormalize
    # x = glm.normalize(c0)
    # # make y orthogonal to x
    # z = glm.normalize(glm.cross(x, c1))
    # y = glm.cross(z, x)
    #
    # # build a column-major 3×3 matrix
    # # glm.mat3(v0, v1, v2) treats v* as *columns*
    # M = glm.mat3(x, y, z)
    #
    # # cast back to quaternion
    # q = glm.quat_cast(M)
    #
    # # extract Euler angles (in radians) in the same convention
    # # glm.eulerAngles returns a vec3 of (pitch, yaw, roll) in radians,
    # # but ordering depends on your GLM version. We’ll assume it matches `order`.
    # e = glm.eulerAngles(q)
    #
    # # convert to degrees
    # return [math.degrees(e.x), math.degrees(e.y), math.degrees(e.z)]
    #

    mat = transforms.rotation_6d_to_matrix(rot6d)
    euler = transforms.matrix_to_euler_angles(mat, order.upper())
    euler = torch.rad2deg(euler)
    return euler


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
        if "Site" in self.name or self.channels is None:
            return
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
        return EulerTo6D(self.rotOrder, rot)

    def setLocalRotQuat(self, q):
        # euler = QuatToEuler(self.rotOrder, q)

        mat = transforms.quaternion_to_matrix(q)

        euler = transforms.matrix_to_euler_angles(mat, self.rotOrder.upper())

        euler = torch.rad2deg(euler)

        # if math.isnan(euler[0]) or math.isnan(euler[1]) or math.isnan(euler[2]):
        #     euler =  torch.zeros(3)

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

        mat = transforms.rotation_6d_to_matrix(rot_6d)
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

    def setGlobalRot(self, q_global):
        if self.parent is not None:

            parent_global = self.parent.globalRot()
            q_local = glm.inverse(parent_global) * q_global
        else:
            q_local = q_global

        self.setLocalRotQuat(torch.tensor(q_local))


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
        elif joint.channels is None or "Site" in joint.name:
            line1 = "%sEnd Site" % (indent)
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
                # print(joint.name)
                # print(len(joint.channels))
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

    def assign_rotation(self, joint, degrees, rot_vec):
        for f in range(self.numFrames()):
            self.readFrame(f)

            p = joint.globalPos()

            rotation_matrix = glm.rotate(glm.radians(degrees), glm.vec3(rot_vec))

            rotated_quat = glm.quat_cast(rotation_matrix)
            joint.setLocalRotQuat(torch.tensor(rotated_quat))
            #

            self.writeFrame(f)

    def local_rotation_quat(self, joint, quat):

        rotated_quat = quat * joint.localRotQuat()
        joint.setLocalRotQuat(torch.tensor(rotated_quat))

    def local_rotation(self, joint, degrees, rot_vec):

        rotation_matrix = glm.rotate(glm.radians(degrees), glm.vec3(rot_vec))

        local_quat = joint.localRotQuat()
        rotated_quat = glm.quat_cast(rotation_matrix) * local_quat

        joint.setLocalRotQuat(torch.tensor(rotated_quat))

        # if torch.any(torch.isnan(torch.tensor(joint.localRotQuat()))):
        #     print(local_quat)
        #     print(rotated_quat)
        #

    def local_rotate(self, joint, degrees, rot_vec):
        for f in range(self.numFrames()):
            self.readFrame(f)

            self.local_rotation(joint, degrees, rot_vec)

            if torch.any(torch.isnan(torch.tensor(joint.localRotQuat()))):  # assign the previous rotation
                self.readFrame(f - 1)
                prev_channelvals = [joint.channelvals[joint.channels.index("Xrotation")],
                                    joint.channelvals[joint.channels.index("Yrotation")],
                                    joint.channelvals[joint.channels.index("Zrotation")]]
                self.readFrame(f)

                joint.channelvals[joint.channels.index("Xrotation")] = prev_channelvals[0]
                joint.channelvals[joint.channels.index("Yrotation")] = prev_channelvals[1]
                joint.channelvals[joint.channels.index("Zrotation")] = prev_channelvals[2]

            self.writeFrame(f)

    def change_rotation_order(self, target_order="zyx"):

        for f in range(self.numFrames()):
            self.readFrame(f)

            for joint in self.joints:
                if joint.channels is not None:
                    # keep original order
                    original_order = joint.rotOrder  # E.g., "ZXY"

                    quat = joint.localRotQuat()
                    updated_euler = QuatToEuler(target_order, quat)

                    joint.setRotOrder(target_order)
                    # assumes new rotation order is in effect
                    joint.setLocalRotEuler(updated_euler)

                    # change it back to original so that we can update it for other frames
                    joint.setRotOrder(original_order)

            self.writeFrame(f)

        # Now change all joints' rotation orders
        for joint in self.joints:
            joint.setRotOrder(target_order)

    def insert_joint(self, new_joint_name, new_joint_child, offsets, rots):
        joint = Joint()
        joint.name = new_joint_name

        joint.offset = offsets

        child = self.jointByName(new_joint_child)
        original_parent = child.parent
        joint.parent = original_parent

        joint.channels = rots  # Initialize the channel values.
        joint.rotOrder = joint.parent.rotOrder

        joint.children = [child]

        joint.channelvals = [0, 0, 0]

        # insert joint to the joint list in the correct location
        # find new index
        ind = 0
        val_ind = 0
        for j in self.joints:
            if j.name == new_joint_child:
                break
            ind += 1
            val_ind += len(j.channels)

        # now add extra frames corresponding to this
        for f in range(self.numFrames()):
            self.readFrame(f)
            vals = self.frames[f]
            vals[val_ind:val_ind] = joint.channelvals
            self.frames[f] = vals
            # we need to insert values 0, 0, 0 after the parent's channels
            # self.writeFrame(f)

        # Make updates to the skeleton

        self.joints[ind:ind] = [joint]

        # shift ids
        for j in self.joints:
            if j.id >= ind:
                j.id += 1

        joint.id = ind

        for i in range(len(original_parent.children)):
            if original_parent.children[i].name == new_joint_child:
                original_parent.children[i] = joint  # replace child

        self.jointMap[new_joint_name] = joint

        child.parent = joint

    def insert_frame(self, bvhInsert):
        f = bvhInsert.frames[0]
        self.frames.insert(0, f)

    def find_alignment_rotations(self, bvhTo):

        def find_bounding_boxes(bvh):
            bbMin = []
            bbMax = []
            for f in range(0, bvh.numFrames()):
                bbMin.append([float('inf'), float('inf'), float('inf')])
                bbMax.append([float('-inf'), float('-inf'), float('-inf')])

                bvh.readFrame(f)
                for i in range(bvh.numJoints()):
                    p = bvh.jointById(i).globalPos()
                    for j in range(3):
                        if p[j] < bbMin[f][j]:
                            bbMin[f][j] = p[j]
                        if p[j] > bbMax[f][j]:
                            bbMax[f][j] = p[j]

            return np.array(bbMax) - np.array(bbMin)

        def scale_translation(bvh, scale_factor):
            for f in range(0, bvh.numFrames() - 1):
                bvh.readFrame(f)
                pos1 = bvh.root.globalPos()

                bvh.readFrame(f + 1)
                pos2 = bvh.root.globalPos()

                delta = (np.array(pos2) - np.array(pos1)) * scale_factor[f]

                pos2 = pos2 + delta
                bvh.root.setGlobalPos(list(pos2))
                bvh.writeFrame(f + 1)

                # shift the rest by delta as well
                for f2 in range(f + 2, bvh.numFrames()):
                    bvh.readFrame(f2)
                    pos = bvh.root.globalPos()
                    pos = np.array(pos) + delta
                    bvh.root.setGlobalPos(list(pos))
                    bvh.writeFrame(f2)

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

        def align_limbs(jointFrom, jointTo):

            if jointFrom is not self.root:

                for f in range(0, self.numFrames()):
                    self.readFrame(f)
                    bvhTo.readFrame(f)
                    global_delta = find_rotation_from_to_limb(jointTo, jointFrom)
                    jointTo.parent.setGlobalRot(global_delta * jointTo.parent.globalRot())

                    bvhTo.writeFrame(f)

            for childFrom, childTo in zip(jointFrom.children, jointTo.children):
                align_limbs(childFrom, childTo)

        # make skeletons as much aligned as possible
        self.update_root_translation()  # fixes  root
        self.local_rotate(self.root, 90, [-1, 0, 0])  # flip axes

        # copy cmu T-pose for the duration of animation
        bvhTo.frames[1:] = [copy.deepcopy(bvhTo.frames[0]) for _ in range(self.numFrames() - 1)]

        for f in range(0, self.numFrames()):
            self.readFrame(f)
            bvhTo.root.setGlobalPos(self.root.globalPos())
            bvhTo.writeFrame(f)

        align_limbs(self.root, bvhTo.root)

        # find bounding boxes of self
        bbFrom = find_bounding_boxes(self)
        bbTo = find_bounding_boxes(bvhTo)

        posScale = bbTo / bbFrom

        scale_translation(bvhTo, posScale)
        # for f in range(0, bvhTo.numFrames()):
        #     bvhTo.readFrame(f)
        #     posScale = bbTo[f] / bbFrom[f]

        # scale_translation(bvhTo, posScale)
        # bvhTo.writeFrame(f)

        return

    def update_root_translation(self):
        # For each frame in the source animation, update the root translation.
        # for j in self.joints:

        for f in range(self.numFrames()):
            self.readFrame(f)
            current_pos = self.root.localPos()  # current translation vector
            # new_pos = [c + d for c, d in zip(current_pos, delta_translation)]
            # new_pos = (0,0,0) #(current_pos[2], current_pos[1], current_pos[0]) # replace x and z
            new_pos = (current_pos[0], current_pos[2], -current_pos[1])  # replace x and z  - flip axes
            self.root.setLocalPos(new_pos)
            self.writeFrame(f)

    def match_skeleton_scale(self, bvhTo):
        def calculate_limb_length(joint):
            """Calculates the length of the limb (distance to parent)."""
            if joint.parent is None:
                return 0  # Root has no limb length

            child_pos = glm.vec3(joint.globalPos())
            parent_pos = glm.vec3(joint.parent.globalPos())

            return glm.length(child_pos - parent_pos)  # np.linalg.norm(child_pos - parent_pos)

        def scale_offsets(joint, scale_factor):
            """Scales the offset of a joint."""
            if joint.parent is not None:
                offset = np.array(joint.localPos())
                scaled_offset = offset * scale_factor
                joint.setLocalPos(scaled_offset.tolist())

            # Recursively scale the limbs of bvh1 to match bvh2

        def scale_joint(joint1, joint2):
            # Compute limb lengths
            length_bvh1 = calculate_limb_length(joint1)
            length_bvh2 = calculate_limb_length(joint2)

            # Avoid division by zero
            if length_bvh1 > 0:
                scale_factor = length_bvh2 / length_bvh1
            else:
                scale_factor = 1.0  # No scaling for root

            # Scale the offset of the joint
            scale_offsets(joint1, scale_factor)

            # Recursively scale child joints
            for child1, child2 in zip(joint1.children, joint2.children):
                scale_joint(child1, child2)

            # Start scaling from the root

        scale_joint(self.root, bvhTo.root)

