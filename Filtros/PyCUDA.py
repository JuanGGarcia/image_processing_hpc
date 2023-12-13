#!pip install pycuda

#!nvidia-smi

#!nvcc --version

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Se usa PyCUDA para el filtrado

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from scipy import ndimage

# Código CUDA para convolución
CUDA_KERNEL = """
__global__ void convolve(float *image, float *kernel, float *output, int width, int height, int kernel_width, int kernel_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    float sum = 0;
    int kernel_radius_x = kernel_width / 2;
    int kernel_radius_y = kernel_height / 2;

    for (int i = -kernel_radius_y; i <= kernel_radius_y; i++) {
        for (int j = -kernel_radius_x; j <= kernel_radius_x; j++) {
            int ix = x + j;
            int iy = y + i;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                sum += kernel[(i + kernel_radius_y) * kernel_width + (j + kernel_radius_x)] * image[iy * width + ix];
            }
        }
    }
    output[y * width + x] = sum;
}
"""

# Función que aplica un kernel a una imagen usando PyCUDA
def apply_kernel(kernel_name, kernel, image):
    kernel_width, kernel_height = kernel.shape
    image_height, image_width = image.shape

    # Cargar el kernel de CUDA
    mod = SourceModule(CUDA_KERNEL)
    convolve = mod.get_function("convolve")

    # Crear arrays de GPU
    image_gpu = gpuarray.to_gpu(image.astype(np.float32))
    kernel_gpu = gpuarray.to_gpu(kernel.astype(np.float32))
    output_gpu = gpuarray.empty_like(image_gpu)

    # Calcular dimensiones de bloques y grids
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(image.shape[1] / block_size[0])), int(np.ceil(image.shape[0] / block_size[1])))

    # Ejecutar el kernel de CUDA
    convolve(image_gpu, kernel_gpu, output_gpu, np.int32(image.shape[1]), np.int32(image.shape[0]), np.int32(kernel_width), np.int32(kernel_height), block=block_size, grid=grid_size)

    # Recuperar los resultados
    output = output_gpu.get()
    return (kernel_name, output, output.min(), output.max(), output.mean(), output.std())

# Función para procesar los resultados y mostrar las imágenes
def process_results(result):
    kernel_name, filtered_image, min_val, max_val, mean_val, std_dev = result
    print(f"Kernel: {kernel_name}")
    print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std Dev: {std_dev}")
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f"Kernel: {kernel_name}")
    plt.show()

# Función que maneja la creación de procesos para cada kernel
def process_image_with_kernels(image_path, kernels):
    # Cargar la imagen
    image = imread(image_path, pilmode='L')

    # Crear un pool de procesos
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Asignar un kernel y la imagen a cada proceso
        results = [pool.apply_async(apply_kernel, args=((k_name, k), image)) for k_name, k in kernels]

        # Obtener y procesar los resultados
        for res in results:
            process_results(res.get())

if __name__ == '__main__':
    # Ruta a la imagen
    image_path = './majors transparent.jpg'

    # Cargar la imagen una sola vez fuera de la función de procesamiento
    image = imread(image_path, pilmode='L')

    # Definir los kernels
    kernel_class_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel_class_2 = np.array([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]])
    kernel_class_3 = np.array([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]])
    kernel_square_3x3 = np.ones((3, 3))  # Ejemplo de un kernel cuadrado 3x3
    kernel_edge_3x3 = np.array([[ 1,  0, -1],[ 0,  0,  0],[-1,  0,  1]])
    kernel_square_5x5 = np.ones((5, 5))  # Ejemplo de un kernel cuadrado 5x5
    kernel_edge_5x5 = np.array([[ 2,  1,  0, -1, -2],
    [ 1,  1,  0, -1, -1],
    [ 0,  0,  0,  0,  0],
    [-1, -1,  0,  1,  1],
    [-2, -1,  0,  1,  2]
    ])

     # Generar los kernels que dependen de la imagen
    kernel_laplace = ndimage.laplace(image)
    kernel_prewitt_vertical = ndimage.prewitt(image, axis=0)
    kernel_prewitt_horizontal = ndimage.prewitt(image, axis=1)


    # Lista de kernels para aplicar a la imagen
    kernels = [kernel_class_1, kernel_class_2, kernel_class_3, kernel_square_3x3,
            kernel_edge_3x3, kernel_square_5x5, kernel_edge_5x5, kernel_sobel_vertical,
            kernel_laplace, kernel_prewitt_horizontal]  # Asegúrate de que todos los kernels estén definidos aquí

    # Lista de kernels para aplicar a la imagen con sus nombres
    kernels = [
        ('class_1', kernel_class_1),
        ('class_2', kernel_class_2),
        ('class_3', kernel_class_3),
        ('kernel_square_3x3', kernel_square_3x3),
        ('kernel_edge_3x3', kernel_edge_3x3),
        ('kernel_square_5x5', kernel_square_5x5),
        ('kernel_edge_5x5', kernel_edge_5x5),
        ('prewitt_vertical', kernel_prewitt_vertical),
        ('prewitt_horizontal', kernel_prewitt_horizontal),
        ('laplace', kernel_laplace)
    ]

# Procesar la imagen con todos los kernels
for kernel_name, kernel in kernels:
    result = apply_kernel(kernel_name, kernel, image)
    process_results(result)