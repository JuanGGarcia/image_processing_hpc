# Se usa multiprocessing para el filtrado

import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import imageio.v2 as imageio
import multiprocessing as mp
import matplotlib.pyplot as plt

class ImageMultiprocessing:
    def __init__(self, image_path):
        self.image = imageio.imread(image_path, pilmode='L')
        self.kernels = self.define_kernels()

    def define_kernels(self):
         # Definir los kernels
        kernel_class_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        kernel_class_2 = np.array([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]])
        kernel_class_3 = np.array([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]])
        kernel_square_3x3 = np.ones((3, 3))  # Ejemplo de un kernel cuadrado 3x3
        kernel_edge_3x3 = np.array([[ 1,  0, -1],
        [ 0,  0,  0],
        [-1,  0,  1]
        ])
        kernel_square_5x5 = np.ones((5, 5))  # Ejemplo de un kernel cuadrado 5x5
        kernel_edge_5x5 = np.array([[ 2,  1,  0, -1, -2],
        [ 1,  1,  0, -1, -1],
        [ 0,  0,  0,  0,  0],
        [-1, -1,  0,  1,  1],
        [-2, -1,  0,  1,  2]
        ])

        # Generar los kernels que dependen de la imagen
        kernel_laplace = ndimage.laplace(self.image)
        kernel_prewitt_vertical = ndimage.prewitt(self.image, axis=0)
        kernel_prewitt_horizontal = ndimage.prewitt(self.image, axis=1)

        # Lista de kernels para aplicar a la imagen con sus nombres

        return [
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
            
    # Función que aplica un kernel a una imagen y guarda las estadísticas
    def apply_kernel(self, kernel_data, image):
        kernel_name, kernel = kernel_data
        filtered_image = convolve2d(image, kernel, mode='same', boundary='wrap')
        return (kernel_name, filtered_image, filtered_image.min(), filtered_image.max(),
                filtered_image.mean(), filtered_image.std())

    # Función para procesar los resultados y mostrar las imágenes
    def process_results(self, result):
        kernel_name, filtered_image, min_val, max_val, mean_val, std_dev = result
        print(f"Kernel: {kernel_name}")
        print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std Dev: {std_dev}")
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f"Kernel: {kernel_name}")
        plt.show()

    def process_image_with_kernels(self):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = [pool.apply_async(self.apply_kernel, args=(k, self.image)) for k in self.kernels]
            for res in results:
                self.process_results(res.get())

if __name__ == '__main__':
    image_path = './majors transparent.jpg'
    processing_instance = ImageMultiprocessing(image_path)
    processing_instance.process_image_with_kernels()                                                                                            


        