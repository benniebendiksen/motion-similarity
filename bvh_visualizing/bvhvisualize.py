# bvhvisualizer.py, Aline Normoyle, 2024

import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def BVHVisualizeFrame(bvh, frame):
    bvh.readFrame(frame)

    # Visualize global locations of joints
    num = bvh.numJoints()
    x = np.zeros(num)
    y = np.zeros(num)
    z = np.zeros(num)
    for i in range(num):
        p = bvh.jointById(i).globalPos()
        x[i] = p[0]
        y[i] = p[1]
        z[i] = p[2]

    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(vertical_axis='y' ) #, share=True)
    ax.scatter(x, y, z, color="g", edgecolor="k")
    ax.set_title("Frame %d"%frame)
    plt.show()

class BVHAnimator:
    def __init__(self, bvh):
        self.bvh = bvh
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')


        #expand bounding boxes a bit for plotting

        # Find bbmin
        bbMin = [float('inf'), float('inf'), float('inf')]
        bbMax = [float('-inf'), float('-inf'), float('-inf')]

        for f in range(bvh.numFrames()):
            bvh.readFrame(f)
            for i in range(bvh.numJoints()):
                p = bvh.jointById(i).globalPos()
                for j in range(3):
                    if p[j] < bbMin[j]:
                        bbMin[j] = p[j]
                    if p[j]> bbMax[j] :
                        bbMax[j] = p[j]

        self.bbMin = np.array(bbMin) - np.array([2, 2, 2])
        self.bbMax = np.array(bbMax) + np.array([2, 2, 2])

        # self.ani = animation.FuncAnimation(self.fig, self.update, interval=bvh.frameRate, frames = range(bvh.numFrames()), init_func=self.setup_plot, blit=False)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=33, frames=range(bvh.numFrames()), init_func=self.setup_plot, blit=False)
        # plt.ion()
        plt.show()

    def read_frame(self, f):
        self.bvh.readFrame(f)
        num = self.bvh.numJoints()
        x = np.zeros(num)
        y = np.zeros(num)
        z = np.zeros(num)
        for i in range(num):
            p = self.bvh.jointById(i).globalPos()
            x[i] = p[0]
            y[i] = p[1]
            z[i] = p[2]
        return x, y, z

    def setup_plot(self):
        x,y,z = self.read_frame(0)
        num_points = self.bvh.numJoints()
        colors = ['r'] * len(x)
        # colors = [
        #     'black',  # Hips
        #     'pink', 'pink', 'pink', 'pink', 'gray',  # Left leg hierarchy
        #     'blue', 'blue', 'blue', 'blue', 'gray',  # Right leg hierarchy
        #     'yellow', 'yellow', 'yellow', 'orange', 'orange', 'gray',  # Spine and neck hierarchy
        #     'red', 'red', 'red', 'red',  # Left arm hierarchy
        #     'red', 'red', 'gray',  # Left hand index and site
        #     'green', 'green', 'green', 'green',  # Right arm hierarchy
        #     'green', 'green', 'gray'  # Right hand index and site
        # ]

        self.scat = self.ax.scatter(x, y, z, color=colors, edgecolor="k")
        self.ax.view_init(vertical_axis='z')

        # Reverse Z-axis to match Unity's forward direction
        # self.ax.invert_zaxis()

        self.plot = self.ax.plot([self.bbMin[0], self.bbMax[0]],[self.bbMin[1], self.bbMin[1]] ,[self.bbMin[2], self.bbMin[2]] , color='r')
        self.plot = self.ax.plot([self.bbMin[0],self.bbMin[0]], [self.bbMin[1], self.bbMax[1]], [self.bbMin[2], self.bbMin[2]], color='g')
        self.plot = self.ax.plot([self.bbMin[0],self.bbMin[0]], [self.bbMin[1], self.bbMin[1]], [self.bbMin[2], self.bbMax[2]], color='b')


        # self.ax.set_xlim(self.bbMin[0], self.bbMax[0])
        # self.ax.set_ylim(self.bbMin[1], self.bbMax[1])
        # self.ax.set_zlim(self.bbMin[2], self.bbMax[2])

        self.ax.set_xlim(self.bbMin[0], self.bbMax[0])
        self.ax.set_ylim(self.bbMin[1], self.bbMax[1])
        self.ax.set_zlim(self.bbMin[2], self.bbMax[2])

        # Set a camera view similar to Unity (adjust azimuth and elevation)
        # self.ax.view_init(elev=3, azim=-80)  # Azim = -90 to make Z point "forward"

        # print(self.bbMin)
        # print(self.bbMax)
        self.title = self.ax.set_title('Frame 0')


        return self.scat,

    def update(self, i):
        idx = min(i, self.bvh.numFrames()-1)
        x,y,z = self.read_frame(idx)
        self.title.set_text("Frame %d"%idx)
        self.scat._offsets3d = (x,y,z)
        # print(f"{x} {y} {z}")
        return self.scat,


class DualBVHAnimator:
    def __init__(self, bvh1, bvh2, distance=None):
        """
        Create a side-by-side visualization of two BVH animations.

        Args:
            bvh1: First BVH object
            bvh2: Second BVH object
            distance: Optional distance metric to display
        """
        self.bvh1 = bvh1
        self.bvh2 = bvh2
        self.distance = distance

        # Create a figure with two subplots side by side
        self.fig = plt.figure(figsize=(12, 6))

        # Add title with distance if provided
        if distance is not None:
            self.fig.suptitle(f"Embedding Distance: {distance:.4f}", fontsize=14)

        # Create 3D subplot for first animation
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title("Animation 1")

        # Create 3D subplot for second animation
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title("Animation 2")

        # Calculate combined bounding box for consistent views
        self.bbMin, self.bbMax = self._calculate_combined_bb()

        # Calculate total frame count (use the minimum of both animations)
        self.total_frames = min(bvh1.numFrames(), bvh2.numFrames())

        # Setup animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update, interval=33,
            frames=range(self.total_frames),
            init_func=self.setup_plot, blit=False
        )

        plt.tight_layout()
        plt.show()

    def _calculate_combined_bb(self):
        """Calculate combined bounding box for both animations"""
        bbMin = [float('inf'), float('inf'), float('inf')]
        bbMax = [float('-inf'), float('-inf'), float('-inf')]

        # Check both animations
        for bvh in [self.bvh1, self.bvh2]:
            for f in range(bvh.numFrames()):
                bvh.readFrame(f)
                for i in range(bvh.numJoints()):
                    p = bvh.jointById(i).globalPos()
                    for j in range(3):
                        if p[j] < bbMin[j]:
                            bbMin[j] = p[j]
                        if p[j] > bbMax[j]:
                            bbMax[j] = p[j]

        # Add padding
        bbMin = np.array(bbMin) - np.array([2, 2, 2])
        bbMax = np.array(bbMax) + np.array([2, 2, 2])

        return bbMin, bbMax

    def read_frame(self, bvh, frame_idx):
        """Read a specific frame from a BVH file"""
        bvh.readFrame(frame_idx)
        num = bvh.numJoints()
        x = np.zeros(num)
        y = np.zeros(num)
        z = np.zeros(num)

        for i in range(num):
            p = bvh.jointById(i).globalPos()
            x[i] = p[0]
            y[i] = p[1]
            z[i] = p[2]

        return x, y, z

    def setup_plot(self):
        """Set up the initial plot"""
        # First animation
        x1, y1, z1 = self.read_frame(self.bvh1, 0)
        colors = ['r'] * len(x1)
        self.scat1 = self.ax1.scatter(x1, y1, z1, color=colors, edgecolor="k")

        # Second animation
        x2, y2, z2 = self.read_frame(self.bvh2, 0)
        colors = ['b'] * len(x2)  # Use different color for second animation
        self.scat2 = self.ax2.scatter(x2, y2, z2, color=colors, edgecolor="k")

        # Set common view settings for both subplots
        for ax in [self.ax1, self.ax2]:
            ax.view_init(vertical_axis='z')
            ax.set_xlim(self.bbMin[0], self.bbMax[0])
            ax.set_ylim(self.bbMin[1], self.bbMax[1])
            ax.set_zlim(self.bbMin[2], self.bbMax[2])

            # Add reference axes
            ax.plot([self.bbMin[0], self.bbMax[0]], [self.bbMin[1], self.bbMin[1]], [self.bbMin[2], self.bbMin[2]],
                    color='r')
            ax.plot([self.bbMin[0], self.bbMin[0]], [self.bbMin[1], self.bbMax[1]], [self.bbMin[2], self.bbMin[2]],
                    color='g')
            ax.plot([self.bbMin[0], self.bbMin[0]], [self.bbMin[1], self.bbMin[1]], [self.bbMin[2], self.bbMax[2]],
                    color='b')

        # Add frame counter
        self.title = self.fig.text(0.5, 0.01, "Frame 0", ha='center')

        return self.scat1, self.scat2

    def update(self, i):
        """Update both animations for the given frame index"""
        idx = min(i, self.total_frames - 1)

        # Update first animation
        x1, y1, z1 = self.read_frame(self.bvh1, idx)
        self.scat1._offsets3d = (x1, y1, z1)

        # Update second animation
        x2, y2, z2 = self.read_frame(self.bvh2, idx)
        self.scat2._offsets3d = (x2, y2, z2)

        # Update frame counter
        self.title.set_text(f"Frame {idx}")

        return self.scat1, self.scat2

