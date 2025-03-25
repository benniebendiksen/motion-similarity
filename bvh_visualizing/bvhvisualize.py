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

