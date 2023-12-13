# Se usa multiprocessing para el filtrado
import imageio
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image  # Import PIL for image conversion
import time

# Define a standalone function for multiprocessing
def apply_kernel_standalone(kernel_data, image):
    kernel_name, kernel = kernel_data
    filtered_image = convolve2d(image, kernel, mode='same', boundary='wrap')
    return (kernel_name, filtered_image, filtered_image.min(), filtered_image.max(),
            filtered_image.mean(), filtered_image.std())

class ImageTemplate:
    def __init__(self, image_path):
        self.image = imageio.imread(image_path, pilmode='L')
        self.kernels = self.define_kernels()

    def run_serial(self, image_paths):
        start_time = time.time()
        results = []
        for path in image_paths:
            image = imageio.imread(path, pilmode='L')
            for kernel_name, kernel in self.kernels.items():
                result = self.apply_kernel((kernel_name, kernel), image)
                processed_result = self.process_results(result)
                results.append(processed_result)
        end_time = time.time()
        return results, end_time - start_time

    def run_parallel(self, image_paths):
        start_time = time.time()
        results = []
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for path in image_paths:
                image = imageio.imread(path, pilmode='L')
                async_results = [pool.apply_async(self.apply_kernel, args=((kernel_name, kernel), image)) for kernel_name, kernel in self.kernels.items()]
                for async_result in async_results:
                    result = async_result.get()
                    processed_result = self.process_results(result)
                    results.append(processed_result)
        end_time = time.time()
        return results, end_time - start_time

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
        kernel_laplace = ndimage.laplace(self.image)  # La función de scipy para Laplace
        kernel_prewitt_vertical = ndimage.prewitt(self.image, axis=0)  # Prewitt vertical
        kernel_prewitt_horizontal = ndimage.prewitt(self.image, axis=1)  # Prewitt horizontal

        # Aplicar el kernel a la imagen
        filtered_image_class_1 = convolve2d(self.image, kernel_class_1, mode='same', boundary='wrap')

        # Calcular las estadísticas solicitadas
        min_val = filtered_image_class_1.min()
        max_val = filtered_image_class_1.max()
        mean_val = filtered_image_class_1.mean()
        std_dev = filtered_image_class_1.std()

        # Mostrar las estadísticas
        print(f"Dimensiones: {filtered_image_class_1.shape}")
        print(f"Valor mínimo: {min_val}")
        print(f"Valor máximo: {max_val}")
        print(f"Valor medio: {mean_val}")
        print(f"Desviación estándar: {std_dev}")

        # Para visualizar la imagen filtrada (opcional)
        plt.imshow(filtered_image_class_1, cmap='gray')
        plt.show()

        return {
            'class_1': kernel_class_1,
            'class_2': kernel_class_2,
            'class_3': kernel_class_3,
            'kernel_square_3x3': kernel_square_3x3,
            'kernel_edge_3x3': kernel_edge_3x3,
            'kernel_square_5x5': kernel_square_5x5,
            'kernel_edge_5x5': kernel_edge_5x5,
            'prewitt_vertical': kernel_prewitt_vertical,
            'prewitt_horizontal': kernel_prewitt_horizontal,
            'laplace': kernel_laplace
        }

    # Función que aplica un kernel a una imagen y guarda las estadísticas
    def apply_kernel(self, kernel_data, image):
        kernel_name, kernel = kernel_data
        filtered_image = convolve2d(image, kernel, mode='same', boundary='wrap')
        return (kernel_name, filtered_image, filtered_image.min(), filtered_image.max(),
                filtered_image.mean(), filtered_image.std())

    # Función para procesar los resultados y mostrar las imágenes
    def process_results(self, result):
        kernel_name, filtered_image, min_val, max_val, mean_val, std_dev = result
        # Convertir a formato adecuado para Streamlit (por ejemplo, a una imagen PIL)
        display_image = Image.fromarray(filtered_image.astype('uint8'), 'L')
        return {
            "kernel": kernel_name,
            "image": display_image,
            "min_val": min_val,
            "max_val": max_val,
            "mean_val": mean_val,
            "std_dev": std_dev
        }

    def run_multiple_images(self, image_paths, selected_kernel_name):
        results = []
        for path in image_paths:
            self.image = imageio.imread(path, pilmode='L')
            selected_kernel = next((k for k_name, k in self.kernels if k_name == selected_kernel_name), None)
            if selected_kernel is None:
                raise ValueError(f"Kernel '{selected_kernel_name}' not found")
            result = self.apply_kernel((selected_kernel_name, selected_kernel), self.image)
            processed_result = self.process_results(result)
            results.append(processed_result)
        return results

    # Función que maneja la creación de procesos para cada kernel
    def process_image_with_kernels(self):
        processed_results = []
        with mp.Pool(processes=mp.cpu_count()) as pool:
            async_results = [pool.apply_async(self.apply_kernel_wrapper, args=(k,)) for k in self.kernels]
            for async_result in async_results:
                result = async_result.get()
                processed_result = self.process_results(result)
                processed_results.append(processed_result)
        return processed_results

    def apply_kernel_wrapper(self, kernel_data):
        return self.apply_kernel(*kernel_data)  # Pass the kernel_data argument correctly
    
    def run(self, selected_kernel_name, image_paths):
        results = []
        for image_path in image_paths:
            # Load the image
            self.image = imageio.imread(image_path, pilmode='L')
            # Get the selected kernel from the dictionary
            selected_kernel = self.kernels.get(selected_kernel_name)
            if selected_kernel is None:
                raise ValueError(f"Kernel '{selected_kernel_name}' not found")
            # Process the image with the selected kernel
            result = self.apply_kernel((selected_kernel_name, selected_kernel), self.image)
            processed_result = self.process_results(result)
            results.append(processed_result)
        return results


if __name__ == '__main__':
    # Define the name of the kernel you want to use
    selected_kernel_name = "class_1"
    # Create an instance of the ImageTemplate class
    image_template = ImageTemplate('processed_images\majors transparent.jpg')

    # Call the run method on this instance
    image_template.run(selected_kernel_name)