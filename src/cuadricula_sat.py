import json
import os
from PIL import Image
import torch
from utils import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile as tiff

import sys
sys.path.append('/home/msiau/workspace/ibeltran/111/SRCNN')
from model import SRCNN
sys.path.append('/home/msiau/workspace/ibeltran/111/SRGAN')
from models import SRResNet, Generator, Discriminator

# data
upscale_factor = 4

# paths
path_json = "/home/msiau/workspace/ibeltran/111/json_sat/sen2venus_images.json"
path_checkpoint_srcnn = "/home/msiau/workspace/ibeltran/111/SRCNN/models/srcnn_sat.pth.tar"
path_checkpoint_srresnet = "/home/msiau/workspace/ibeltran/111/SRGAN/models/checkpoint_srresnet_sat.pth.tar"
path_checkpoint_srgan = "/home/msiau/workspace/ibeltran/111/SRGAN/models/checkpoint_srgan_sat.pth.tar"
path_to_save = "/home/msiau/data/tmp/ibeltran/sat/sen2venus_SR/res_sat_rgb/"

# load models
srcnn_model = torch.load(path_checkpoint_srcnn)["model"]
srcnn_model.eval()
srresnet_model = torch.load(path_checkpoint_srresnet)["model"]
srresnet_model.eval()
srgan_model = torch.load(path_checkpoint_srgan)["generator"]
srgan_model.eval()

models = [srcnn_model, srresnet_model, srgan_model]
titles_images = ["GT", "bicubic", "SRCNN", "SRResNet", "SRGAN"]

def load_image(image_type, path):
    if image_type == "rgb":
        img = Image.open(path, mode='r')
        img = img.convert('RGB')
        
    elif image_type == "rgb_tensor" or image_type == "rgbn_tensor":
        tensor = torch.load(path)
        tensor = tensor.float() / 10000.0
        # get only bgr channels and the first patch
        blue, green, red = tensor[0, 0, :, :], tensor[0, 1, :, :], tensor[0, 2, :, :]
        if image_type == "rgb_tensor":
            rgb_tensor = torch.stack([red, green, blue], dim=0) 
        if image_type == "rgbn_tensor":
            infrared = tensor[0, 3, :, :]
            rgb_tensor = torch.stack([red, green, blue, infrared], dim=0)
        rgb_image = rgb_tensor.squeeze().permute(1, 2, 0).numpy()
        rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_image)

    elif image_type == "rgbn":
        img = tiff.imread(path)
        # Normalize the 16-bit image data to the range [0, 1]
        img = img.astype(np.float32) / np.max(img)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

    return img

def resolve(name_image, models):
    images_to_save = []
    # load lr and hr images
    hr_image = load_image("rgb_tensor", name_image)
    # hr_image = Image.open(name_image, mode='r').convert('RGB')
    images_to_save.append(hr_image)
    lr_image = hr_image.resize((int(hr_image.width / upscale_factor), int(hr_image.height / upscale_factor)),
                           Image.BICUBIC)
    # bicubic upsampling
    bicubic_image = lr_image.resize((hr_image.width, hr_image.height), Image.BICUBIC)
    images_to_save.append(bicubic_image)
    
    for idx, model in enumerate(models):
        # Super-resolution (SR) 
        if idx == 0:
            sr_image = model(convert_image(bicubic_image, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        elif idx == 1:
            sr_image = model(convert_image(lr_image, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        elif idx == 2:
            sr_image = model(convert_image(lr_image, source='pil', target='[-1, 1]').unsqueeze(0).to(device))
            
        sr_image = sr_image.squeeze(0).cpu().detach()
        sr_image = convert_image(sr_image, source='[-1, 1]', target='pil')
        images_to_save.append(sr_image)
    
    return images_to_save

    
    
if __name__ == "__main__":
    with open(path_json, 'r') as f:
        data_json = json.load(f)

    for path_img in data_json:
        if os.path.exists(path_img):
            file_name = os.path.basename(path_img)
            name, _ = file_name.split(".")
            
            images_to_save = resolve(path_img, models)
            
                
            
            fig, axs = plt.subplots(nrows=1, ncols=len(titles_images))
            # fig, axs = plt.subplots(nrows=1, ncols=len(titles_images), figsize=(15, 10))
            
            for i, img in enumerate(images_to_save):
                axs[i].set_title(titles_images[i])
                if np.array(img).shape[2] > 3:
                    axs[i].imshow(np.array(img)[50:400,50:400,:-1])
                else:
                    axs[i].imshow(np.array(img)[50:400,50:400])
                axs[i].axis('off')
            
            # plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02, hspace=0.02)
            plt.rcParams.update({'font.size': 8})
            plt.tight_layout(pad=0)
            plt.savefig(path_to_save + name, bbox_inches='tight', pad_inches=0)
            plt.close()

            
            