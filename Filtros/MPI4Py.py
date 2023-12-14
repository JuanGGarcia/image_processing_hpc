import os
from matplotlib import pyplot as plt
from mpi4py import MPI
import imageio.v2 as imageio
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from PIL import Image
import time

class ImageMPI:
    def __init__(self, image_path):
        self.image_path = image_path
        self.kernels = self.define_kernels()

    def run(self, selected_kernel_name, image_paths):
        # Método para procesar múltiples imágenes con el kernel seleccionado
        results = []
        for image_path in image_paths:
            # Actualizar la ruta de la imagen y ejecutar el procesamiento en paralelo
            self.image_path = image_path
            processed_results = self.run_parallel_mpi()

            # Filtrar los resultados por el kernel seleccionado
            for result in processed_results:
                if result['kernel'] == selected_kernel_name:
                    results.append(result)

        return results

    def define_kernels(self):
        # Definir los kernels
        kernel_class_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        kernel_class_2 = np.array([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]])
        kernel_class_3 = np.array([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]])
        kernel_square_3x3 = np.ones((3, 3))  # kernel cuadrado 3x3
        kernel_edge_3x3 = np.array([[ 1,  0, -1],[ 0,  0,  0],[-1,  0,  1]
        ])
        kernel_square_5x5 = np.ones((5, 5))  # kernel cuadrado 5x5
        kernel_edge_5x5 = np.array([
        [ 2,  1,  0, -1, -2],
        [ 1,  1,  0, -1, -1],
        [ 0,  0,  0,  0,  0],
        [-1, -1,  0,  1,  1],
        [-2, -1,  0,  1,  2]
        ])
        #kernel_laplace = ndimage.laplace(self.image)  # La función de scipy para Laplace
        #kernel_prewitt_vertical = ndimage.prewitt(self.image, axis=0)  # Prewitt vertical
        #kernel_prewitt_horizontal = ndimage.prewitt(self.image, axis=1)  # Prewitt horizontal

        # Aplicar el kernel a la imagen
        #filtered_image_class_1 = convolve2d(self.image, kernel_class_1, mode='same', boundary='wrap')

        # Calcular las estadísticas solicitadas
        #min_val = filtered_image_class_1.min()
        #max_val = filtered_image_class_1.max()
        #mean_val = filtered_image_class_1.mean()
        #std_dev = filtered_image_class_1.std()

        # Mostrar las estadísticas
        #print(f"Dimensiones: {filtered_image_class_1.shape}")
        #print(f"Valor mínimo: {min_val}")
        #print(f"Valor máximo: {max_val}")
        #print(f"Valor medio: {mean_val}")
        #print(f"Desviación estándar: {std_dev}")

        # Para visualizar la imagen filtrada (opcional)
        #plt.imshow(filtered_image_class_1, cmap='gray')
        plt.show()

        return {
            'class_1': kernel_class_1,
            'class_2': kernel_class_2,
            'class_3': kernel_class_3,
            'kernel_square_3x3': kernel_square_3x3,
            'kernel_edge_3x3': kernel_edge_3x3,
            'kernel_square_5x5': kernel_square_5x5,
            'kernel_edge_5x5': kernel_edge_5x5,
            #'prewitt_vertical': kernel_prewitt_vertical,
            #'prewitt_horizontal': kernel_prewitt_horizontal,
            #'laplace': kernel_laplace
        }
    
    def apply_dynamic_kernels(self, image):
        # Aplicar kernels que dependen de la imagen específica
        kernel_laplace = ndimage.laplace(image)
        kernel_prewitt_vertical = ndimage.prewitt(image, axis=0)
        kernel_prewitt_horizontal = ndimage.prewitt(image, axis=1)

        dynamic_kernels = {
            'laplace': kernel_laplace,
            'prewitt_vertical': kernel_prewitt_vertical,
            'prewitt_horizontal': kernel_prewitt_horizontal
        }

        return dynamic_kernels

    def apply_kernel_standalone(self, kernel_data, image):
        kernel_name, kernel = kernel_data

        # Agregar verificación y depuración
        if not isinstance(image, np.ndarray) or len(image.shape) != 2:
            raise ValueError(f"La imagen debe ser una matriz numpy 2-D. Recibido: {type(image)} con forma {image.shape}")

        if not isinstance(kernel, np.ndarray) or len(kernel.shape) != 2:
            raise ValueError(f"El kernel debe ser una matriz numpy 2-D. Recibido: {type(kernel)} con forma {kernel.shape}")

        # Verificar si el kernel es una matriz numpy
        if not isinstance(kernel, np.ndarray) or kernel.ndim != 2:
            raise TypeError(f"El kernel '{kernel_name}' debe ser una matriz numpy 2-D, pero recibido: {type(kernel)}")


        filtered_image = convolve2d(image, kernel, mode='same', boundary='wrap')
        return (kernel_name, filtered_image, filtered_image.min(), filtered_image.max(), filtered_image.mean(), filtered_image.std())


    def process_results(self, result):
        kernel_name, filtered_image, min_val, max_val, mean_val, std_dev = result
        display_image = Image.fromarray(filtered_image.astype('uint8'), 'L')
        return {
            "kernel": kernel_name,
            "image": display_image,
            "min_val": min_val,
            "max_val": max_val,
            "mean_val": mean_val,
            "std_dev": std_dev
        }

    def run_parallel_mpi(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            # Validar y cargar la imagen
            if not os.path.exists(self.image_path):
                raise FileNotFoundError(f"No se encontró el archivo: '{self.image_path}'")
            image = imageio.imread(self.image_path, pilmode='L')

            # Dividir los nombres de los kernels para distribuir entre los procesos
            kernel_names = list(self.kernels.keys())
            split_indices = np.array_split(range(len(kernel_names)), size)
        else:
            image = None
            split_indices = None

        local_indices = comm.scatter(split_indices, root=0)
        image = comm.bcast(image, root=0)

        # Procesar los kernels asignados a este proceso
        results = []
        for idx in local_indices:
            kernel_name = kernel_names[idx]
            kernel = self.kernels[kernel_name]
            result = self.apply_kernel_standalone((kernel_name, kernel), image)
            processed_result = self.process_results(result)
            results.append(processed_result)

        # Recolectar y combinar resultados
        gathered_results = comm.gather(results, root=0)

        if rank == 0:
            final_results = [item for sublist in gathered_results for item in sublist]
            return final_results


def main():
    image_path = '../processed_images'
    kernels = {
        'class_1': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        'class_2': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        'class_3': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        'kernel_square_3x3': np.ones((3, 3)),
        'kernel_edge_3x3': np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]),
        'kernel_square_5x5': np.ones((5, 5)),
        'kernel_edge_5x5': np.array([[2, 1, 0, -1, -2], [1, 1, 0, -1, -1], [0, 0, 0, 0, 0], [-1, -1, 0, 1, 1], [-2, -1, 0, 1, 2]])
    }
    processor = ImageMPI(image_path, kernels)
    results = processor.run_parallel_mpi()

    if results:
        for result in results:
            print(f"Kernel: {result['kernel']}, Min: {result['min_val']}, Max: {result['max_val']}, Mean: {result['mean_val']}, Std Dev: {result['std_dev']}")
            plt.imshow(result['image'], cmap='gray')
            plt.title(f"Kernel: {result['kernel']}")
            plt.show()

if __name__ == '__main__':
    main()
