import os
import cv2

def downscale_images(input_folder, output_folder, scale_factor=4):
    # Verificar si la carpeta de salida existe, si no, crearla
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtener la lista de archivos en la carpeta de entrada
    files = os.listdir(input_folder)

    for file in files:
        # Leer la imagen de alta resolución
        img_hr = cv2.imread(os.path.join(input_folder, file))

        # Obtener las dimensiones de la imagen original
        height, width, _ = img_hr.shape

        # Escalar la imagen usando INTER_LINEAR para mejor calidad
        img_lr = cv2.resize(img_hr, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_LINEAR)

        # Guardar la imagen de baja resolución en la carpeta de salida
        cv2.imwrite(os.path.join(output_folder, file), img_lr)

    print("El downscale de las imágenes ha sido completado.")

# Ejemplo de uso
path = "/home/msiau/data/tmp/ibeltran/data/urban100/"
input_folder = "HR"
output_folder = "LR"

downscale_images(path + input_folder, path + output_folder)
