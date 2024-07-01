import cv2
import matplotlib.pyplot as plt

# Función para cargar una imagen y aplicar un modelo de super-resolución
def super_resolve_image(image_path, model):
    if model == "GT":
        if image_path == "img_001.png":
            path = "/home/msiau/data/tmp/ibeltran/data/BSDS200/HR/71046.png"
        if image_path == "img_002.png":
            path = "/home/msiau/data/tmp/ibeltran/data/urban100/HR/img_002.png"
        if image_path == "img_003.png":
            path = "/home/msiau/data/tmp/ibeltran/data/General100/HR/im_059.png"
    if model == "bicubic":
        path = "/home/msiau/workspace/ibeltran/111/SRCNN/outputs/bic_" + image_path
    if model == "SRCNN":
        path = "/home/msiau/workspace/ibeltran/111/SRCNN/outputs/" + image_path
    if model == "SRResNet":
        path = "/home/msiau/workspace/ibeltran/111/SRGAN/outputs/resnet_" + image_path
    if model == "SRGAN":
        path = "/home/msiau/workspace/ibeltran/111/SRGAN/outputs/gan_" + image_path
    if model == "SwinIR":
        if image_path == "img_001.png":
            path = "/home/msiau/workspace/ibeltran/111/SwinIR/results/swinir_classical_sr_x4/71046_SwinIR.png"
        if image_path == "img_002.png":
            path = "/home/msiau/workspace/ibeltran/111/SwinIR/results/swinir_classical_sr_x4/img_002_SwinIR.png"
        if image_path == "img_003.png":
            path = "/home/msiau/workspace/ibeltran/111/SwinIR/results/swinir_classical_sr_x4/im_059_SwinIR.png"
    image = cv2.imread(path)
    return image

# Rutas de las imágenes y modelos
image_paths = ["img_001.png", "img_002.png", "img_003.png"]
model_names = ["GT", "bicubic", "SRCNN", "SRResNet", "SRGAN", "SwinIR"]

# Crear una figura para mostrar las imágenes
fig, axs = plt.subplots(nrows=3, ncols=len(model_names), figsize=(15, 10))

# Iterar sobre las imágenes y modelos
for i, image_path in enumerate(image_paths):
  
    for j, model_name in enumerate(model_names):
        # Aplicar el modelo de super-resolución a la imagen
        super_resolved_image = super_resolve_image(image_path, model_name)
        
        # Mostrar la imagen super-resuelta
        if i == 0:
            axs[i, j].set_title(model_name)
        axs[i, j].imshow(cv2.cvtColor(super_resolved_image, cv2.COLOR_BGR2RGB))
        axs[i, j].axis('off')

# Ajustar espaciado y guardar la figura como una imagen
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
plt.tight_layout(pad=0.1)
plt.savefig("resultado_superresolucion.png")
