#pip install numba

# Filtrar con OpenMP
# No funciona
#from numba import jit, prange

import numpy as np
from scipy.signal import convolve2d
from imageio import imread
from mpi4py import MPI
import matplotlib.pyplot as plt

def process_image_with_kernels(image, kernels):
    all_results = []
    for i in prange(len(kernels)):
        kernel_data = kernels[i]
        result = apply_kernel(kernel_data, image)
        all_results.append(result)
    return all_results

def apply_kernel(kernel_data, image):
    kernel_name, kernel = kernel_data
    filtered_image = convolve2d(image, kernel, mode='same', boundary='wrap')
    return (kernel_name, filtered_image, filtered_image.min(), filtered_image.max(),
            filtered_image.mean(), filtered_image.std())

def display_results(all_results):
    for result in all_results:
        kernel_name, filtered_image, min_val, max_val, mean_val, std_dev = result
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f"Kernel: {kernel_name}")
        plt.show()

if __name__ == '__main__':
    image_path = './majors transparent.jpg'
    image = imread(image_path, pilmode='L')
    # Define the kernels
    kernel_class_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel_class_2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_class_3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_square_3x3 = np.ones((3, 3))
    kernel_edge_3x3 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    kernel_square_5x5 = np.ones((5, 5))
    kernel_edge_5x5 = np.array([[2, 1, 0, -1, -2], [1, 1, 0, -1, -1], [0, 0, 0, 0, 0], [-1, -1, 0, 1, 1], [-2, -1, 0, 1, 2]])
    kernel_sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_sobel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Create additional kernels using the image
    kernel_laplace = ndimage.laplace(image)
    kernel_prewitt_vertical = ndimage.prewitt(image, axis=0)
    kernel_prewitt_horizontal = ndimage.prewitt(image, axis=1)

    kernels = [
        ('class_1', kernel_class_1),
        ('class_2', kernel_class_2),
        ('class_3', kernel_class_3),
        ('square_3x3', kernel_square_3x3),
        ('edge_3x3', kernel_edge_3x3),
        ('square_5x5', kernel_square_5x5),
        ('edge_5x5', kernel_edge_5x5),
        ('sobel_vertical', kernel_sobel_vertical),
        ('sobel_horizontal', kernel_sobel_horizontal),
        ('laplace', kernel_laplace),
        ('prewitt_vertical', kernel_prewitt_vertical),
        ('prewitt_horizontal', kernel_prewitt_horizontal)
    ]

    all_results = process_image_with_kernels(image, kernels)
    display_results(all_results)