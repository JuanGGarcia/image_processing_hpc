#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KSIZE 3 // Tamaño del kernel

// Definición de un kernel simple para el ejemplo
float kernel[KSIZE][KSIZE] = {
    {0, -1, 0},
    {-1, 4, -1},
    {0, -1, 0}};

// Función para aplicar el kernel a un punto de la imagen
float apply_kernel(unsigned char *image, int width, int x, int y)
{
    float sum = 0.0;
    for (int i = 0; i < KSIZE; i++)
    {
        for (int j = 0; j < KSIZE; j++)
        {
            int ix = x + i - KSIZE / 2;
            int jy = y + j - KSIZE / 2;
            if (ix >= 0 && ix < width && jy >= 0 && jy < width)
            {
                sum += image[ix * width + jy] * kernel[i][j];
            }
        }
    }
    return sum;
}

int main()
{
    int width, height, channels;
    unsigned char *image = stbi_load("Image_1.jpg", &width, &height, &channels, 1);
    if (image == NULL)
    {
        printf("Error al cargar la imagen\n");
        return 1;
    }

    printf("Imagen cargada correctamente. Dimensiones: %d x %d\n", width, height);

    unsigned char *output_image = malloc(width * height * sizeof(unsigned char));

#pragma omp parallel for
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            output_image[i * width + j] = (unsigned char)apply_kernel(image, width, i, j);
        }
    }

    // Guardar la imagen
    stbi_write_png("output.png", width, height, 1, output_image, width);

    printf("La imagen se ha guardado correctamente.\n");

    // Liberar memoria
    stbi_image_free(image);
    free(output_image);

    return 0;
}