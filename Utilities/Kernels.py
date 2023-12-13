import imageio
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image

class KernelManager:
    def __init__(self, image):
        self.image = image

    def get_kernels(self):
        return [
            ('class_1', self.class_1_kernel()),
            ('class_2', self.class_2_kernel()),
            ('class_3', self.class_3_kernel()),
            ('kernel_square_3x3', self.square_3x3_kernel()),
            ('kernel_edge_3x3', self.edge_3x3_kernel()),
            ('kernel_square_5x5', self.square_5x5_kernel()),
            ('kernel_edge_5x5', self.edge_5x5_kernel()),
            ('prewitt_vertical', self.prewitt_vertical_kernel()),
            ('prewitt_horizontal', self.prewitt_horizontal_kernel()),
            ('laplace', self.laplace_kernel())
        ]

    def class_1_kernel(self):
        return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    def class_2_kernel(self):
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    def class_3_kernel(self):
        return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    def square_3x3_kernel(self):
        return np.ones((3, 3))

    def edge_3x3_kernel(self):
        return np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    def square_5x5_kernel(self):
        return np.ones((5, 5))

    def edge_5x5_kernel(self):
        return np.array([
            [2, 1, 0, -1, -2],
            [1, 1, 0, -1, -1],
            [0, 0, 0, 0, 0],
            [-1, -1, 0, 1, 1],
            [-2, -1, 0, 1, 2]
        ])

    def prewitt_vertical_kernel(self):
        return ndimage.prewitt(self.image, axis=0)

    def prewitt_horizontal_kernel(self):
        return ndimage.prewitt(self.image, axis=1)

    def laplace_kernel(self):
        return ndimage.laplace(self.image)
    


