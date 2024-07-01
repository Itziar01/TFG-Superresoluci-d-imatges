import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import ImageTransforms
import numpy as np
import tifffile as tiff




class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, split, crop_size, scaling_factor, model_type, image_type, lr_img_type, hr_img_type, test_data_name=None):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param model_type: name of the model type
        :param image_type: type of image (rgb, rgb_tensor, tensor)
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """

        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.model_type = model_type
        self.image_type = image_type.lower()
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name

        assert self.split in {'train', 'test'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("Please provide the name of the test dataset!")
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
        # (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        # Read list of image-paths
        if self.split == 'train': 
            if self.image_type == "rgbn":
                with open(os.path.join(data_folder, 'train_rgbn_images.json'), 'r') as j:
                    self.images = json.load(j)
            else:
                with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
                    self.images = json.load(j)
        else:
            try:
                with open(os.path.join(data_folder, self.test_data_name + '_test_images.json'), 'r') as j:
                    self.images = json.load(j)
            except:
                with open(os.path.join(data_folder, self.test_data_name + '_images.json'), 'r') as j:
                    self.images = json.load(j)

        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         model_type=self.model_type,
                                         #image_type=self.image_type,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        if self.image_type == "rgb":
            img = Image.open(self.images[i], mode='r')
            img = img.convert('RGB')
            
        elif self.image_type == "rgb_tensor" or self.image_type == "rgbn_tensor":
            tensor = torch.load(self.images[i])
            tensor = tensor.float() / 10000.0
            # get only bgr channels and the first patch
            blue, green, red = tensor[0, 0, :, :], tensor[0, 1, :, :], tensor[0, 2, :, :]
            if self.image_type == "rgb_tensor":
                rgb_tensor = torch.stack([red, green, blue], dim=0) 
            if self.image_type == "rgbn_tensor":
                infrared = tensor[0, 3, :, :]
                rgb_tensor = torch.stack([red, green, blue, infrared], dim=0)
            rgb_image = rgb_tensor.squeeze().permute(1, 2, 0).numpy()
            rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(rgb_image)

        elif self.image_type == "rgbn":
            img = tiff.imread(self.images[i])
            # Normalize the 16-bit image data to the range [0, 1]
            img = img.astype(np.float32) / np.max(img)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)

        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.images)