
# Filtrar con MPI4Py
from mpi4py import MPI
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import imageio.v2 as imageio
import os

def apply_kernel(kernel_data, image):
    kernel_name, kernel = kernel_data
    filtered_image = convolve2d(image, kernel, mode='same', boundary='wrap')
    return (kernel_name, filtered_image, filtered_image.min(), filtered_image.max(),
            filtered_image.mean(), filtered_image.std())

def save_image(result, output_folder):
    kernel_name, filtered_image, _, _, _, _ = result
    output_path = os.path.join(output_folder, f"{kernel_name}.png")
    imageio.imsave(output_path, np.clip(filtered_image, 0, 255).astype(np.uint8))

def process_image_with_kernels(image, kernels, output_folder):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Ensure output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in range(1, size):
            kernel_data = kernels[i-1] if i-1 < len(kernels) else ('done', None)
            comm.send(kernel_data, dest=i, tag=i)

        for _ in range(1, size):
            result = comm.recv(source=MPI.ANY_SOURCE)
            save_image(result, output_folder)
    else:
        kernel_data = comm.recv(source=0, tag=rank)
        while kernel_data[0] != 'done':
            result = apply_kernel(kernel_data, image)
            comm.send(result, dest=0)
            kernel_data = comm.recv(source=0, tag=rank)

if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Initialize additional_kernels for all ranks
    additional_kernels = []

    # Define kernels and load image only on the master node
    if rank == 0:
        image_path = './majors transparent.jpg'
        image = imageio.imread(image_path, pilmode='L')
       # Definir los kernels
        kernel_class_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        kernel_class_2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        kernel_class_3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_square_3x3 = np.ones((3, 3))
        kernel_edge_3x3 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
        kernel_square_5x5 = np.ones((5, 5))
        kernel_edge_5x5 = np.array([[2, 1, 0, -1, -2], [1, 1, 0, -1, -1], [0, 0, 0, 0, 0], [-1, -1, 0, 1, 1], [-2, -1, 0, 1, 2]])

        kernels = [
            ('class_1', kernel_class_1),
            ('class_2', kernel_class_2),
            ('class_3', kernel_class_3),
            ('square_3x3', kernel_square_3x3),
            ('edge_3x3', kernel_edge_3x3),
            ('square_5x5', kernel_square_5x5),
            ('edge_5x5', kernel_edge_5x5),
            # The following kernels will be added after broadcasting the image
        ]
        output_folder = './processed_images'
        #process_image_with_kernels(image, kernels, output_folder)
    else:
        #process_image_with_kernels(None, None, None)
        kernels = []


    # Broadcast the image to all nodes
    image = comm.bcast(image, root=0)

    # Generate image-dependent kernels after broadcasting the image
    if rank != 0:  # Only for worker nodes
        kernel_laplace = ndimage.laplace(image)
        kernel_prewitt_vertical = ndimage.prewitt(image, axis=0)
        kernel_prewitt_horizontal = ndimage.prewitt(image, axis=1)
        additional_kernels = [
            ('laplace', kernel_laplace),
            ('prewitt_vertical', kernel_prewitt_vertical),
            ('prewitt_horizontal', kernel_prewitt_horizontal)
        ]
    #else:  # For the master node
        #additional_kernels = []

     # Combine static and dynamic kernels
    all_kernels = kernels + additional_kernels

    # Process the image with kernels using MPI
    process_image_with_kernels(image, all_kernels, output_folder if rank == 0 else None)
