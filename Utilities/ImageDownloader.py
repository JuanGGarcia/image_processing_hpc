import os
import shutil
from bing_image_downloader import downloader
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading

class ImageHandler:
    def clear_directory(self, directory):
        """ Borra todos los archivos y subdirectorios en el directorio especificado. """
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    def move_images_to_directory(self, source_directory, target_directory):
        """ Mueve todas las imágenes de source_directory a target_directory. """
        for filename in os.listdir(source_directory):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                shutil.move(os.path.join(source_directory, filename), target_directory)

    def download_images(self, tema, num_imagenes, output_directory):
        """ Descarga imágenes y luego las mueve a output_directory. """
        downloader.download(tema, limit=num_imagenes, output_dir='temp')
        self.move_images_to_directory(f'temp/{tema}', output_directory)

    def download_task(self, tema, num_imagenes, output_directory):
        thread_id = threading.get_ident()
        print(f"Hilo {thread_id}: Descargando {num_imagenes} imágenes de {tema} en {output_directory}")
        self.download_images(tema, num_imagenes, output_directory)

if __name__ == '__main__':
    temas = ['cats']
    num_imagenes = 5
    output_directory = 'processed_images'
    image_handler = ImageHandler()
    image_handler.clear_directory(output_directory)

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for tema in temas:
            executor.submit(image_handler.download_task, tema, num_imagenes, output_directory)
