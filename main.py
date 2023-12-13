import streamlit as st
import os
import random
from PIL import Image
import multiprocessing
from Filtros.Template import ImageTemplate
from Filtros.Multiprocessing import ImageMultiprocessing
from Utilities.ImageDownloader import ImageHandler
import numpy as np
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor

# App title and headers
st.title("Image Processing App")
st.header("Parallel Computing in Image Processing")
st.write('Explore different parallel computing technologies applied to image processing')

# Display system information
st.write(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
st.write(f"Number of threads available: {os.cpu_count() * 2}")

# Search bar
search_query = st.text_input("Search for images:")

# Define technology options
technologies = {
    "C": "C is a general-purpose, procedural computer programming language supporting structured programming.",
    "OpenMP": "OpenMP (Open Multi-Processing) is an API that supports multi-platform shared-memory multiprocessing programming in C, C++, and Fortran.",
    "Multiprocessing": "Multiprocessing refers to the ability of a system to support more than one processor at the same time. In Python, it's a module that allows you to create processes.",
    "MPI4PY": "MPI for Python (MPI4PY) is a Python package that provides bindings to the Message Passing Interface (MPI) standard designed for high-performance parallel computing.",
    "PyCUDA": "PyCUDA lets you access Nvidia‘s CUDA parallel computation API from Python. It's useful for executing high-performance parallel computing tasks."
}

# Define kernel options
kernels = {
    "class_1": {
        "description": "Class 1 Edge Detection\nMatrix:\n" + str(np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    },
    "class_2": {
        "description": "Class 2 Edge Detection\nMatrix:\n" + str(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    },
    "class_3": {
        "description": "Class 3 Edge Detection\nMatrix:\n" + str(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    },
    "kernel_square_3x3": {
        "description": "3x3 Square Kernel\nMatrix:\n" + str(np.ones((3, 3)))
    },
    "kernel_edge_3x3": {
        "description": "3x3 Edge Detection Kernel\nMatrix:\n" + str(np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]))
    },
    "kernel_square_5x5": {
        "description": "5x5 Square Kernel\nMatrix:\n" + str(np.ones((5, 5)))
    },
    "kernel_edge_5x5": {
        "description": "5x5 Edge Detection Kernel\nMatrix:\n" + str(np.array([[2, 1, 0, -1, -2], [1, 1, 0, -1, -1], [0, 0, 0, 0, 0], [-1, -1, 0, 1, 1], [-2, -1, 0, 1, 2]]))
    },
    "laplace": {
        "description": "Laplace Kernel for Edge Detection\n(Matrix is computed based on the specific image)"
    },
    "prewitt_vertical": {
        "description": "Prewitt Vertical Kernel\n(Matrix is computed based on the specific image)"
    },
    "prewitt_horizontal": {
        "description": "Prewitt Horizontal Kernel\n(Matrix is computed based on the specific image)"
    }
}

# Technology selection
selected_technology = st.selectbox("Select a technology:", list(technologies.keys()))
st.write(technologies[selected_technology])

# Kernel selection
selected_kernel = st.selectbox("Select a kernel:", list(kernels.keys()))
# Display the description of the selected kernel
st.write(kernels[selected_kernel]["description"])

# Image path
#image_path = './majors transparent.jpg'
processed_images_path = 'processed_images'

# Mapping of technology names to their classes
technology_classes = {
    "Multiprocessing": ImageTemplate,
    # "OpenMP": ImageOpenMP,  # Uncomment if OpenMP is to be used
    # "Multiprocessing": ImageMultiprocessing,  
}

# Instancia de ImageHandler
image_handler = ImageHandler()

# Botón para confirmar la selección
if st.button('Choose this Filter', key="filter_selection"):
    st.write(f"You selected: {selected_technology}")
    st.write(technologies[selected_technology])

    if search_query:
        st.write(f"Downloading images for: {search_query}")

        output_directory = 'processed_images'
        num_images = 15  # Número de imágenes a descargar
        image_handler.clear_directory(output_directory)
        # Descargar imágenes
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future = executor.submit(image_handler.download_task, search_query, num_images, output_directory)

        # Espera a que se complete la descarga
        future.result()

        # Procesar imágenes descargadas
        all_image_paths = [os.path.join(output_directory, filename) for filename in os.listdir(output_directory) if filename.endswith(('.jpg', '.png', '.jpeg'))]
        selected_image_paths = random.sample(all_image_paths, min(10, len(all_image_paths)))

        for image_path in selected_image_paths:
            tech_class = technology_classes[selected_technology](image_path)
            results = tech_class.run(selected_kernel, [image_path])

            for result in results:
                col1, col2 = st.columns(2)  # Crea dos columnas
                with col1:  # Primera columna para la imagen original
                    st.image(Image.open(image_path), caption="Original Image", use_column_width=True)
                with col2:  # Segunda columna para la imagen procesada
                    st.image(result['image'], caption=f"Processed Image - Kernel: {result['kernel']}", use_column_width=True)
                    st.write(f"Min: {result['min_val']}, Max: {result['max_val']}, Mean: {result['mean_val']}, Std Dev: {result['std_dev']}")
    else:
        st.write("Please enter a search query.")