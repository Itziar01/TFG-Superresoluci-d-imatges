import os
import json
import torch
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_data_lists(train_folders, test_folders, min_size, output_folder):
    '''
    Create lists for images in the training set and each of the test sets.

    @train_folders: folders containing the training images; these will be merged
    @test_folders: folders containing the test images; each test folder will form its own test set
    @min_size: minimum width and height of images to be considered
    @output_folder: save data lists here
    '''
    print("\nCreating data lists... this may take some time.\n")
    train_images = list()
    for d in train_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("There are %d images in the training data.\n" % len(train_images))
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-2]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print("There are %d images in the %s test data.\n" % (len(test_images), test_name))
        with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
            json.dump(test_images, j)

    print("JSONS containing lists of Train and Test images have been saved to %s\n" % output_folder)
    
def create_data_list_sat(data_folder, min_size, output_folder):
    '''
    Create lists for images in the training set and test set using random split.
    
    @data_folder: folder containing all images
    @min_size: minimum width and height of images to be considered
    @output_folder: save data lists here
    '''
    print("\nCreating data lists... this may take some time.\n")
    all_images = list()

    # Gather all image paths
    for subdir, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                img = Image.open(img_path)
                if img.width >= min_size and img.height >= min_size:
                    all_images.append(img_path)

    # Randomly shuffle the list of image paths
    random.shuffle(all_images)

    # Calculate split indexes
    split_idx = int(0.7 * len(all_images))

    # Split the list into training and test sets
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    # Save training images list to JSON
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    # Save test images list to JSON
    with open(os.path.join(output_folder, 'test_images.json'), 'w') as j:
        json.dump(test_images, j)

    print("JSONs containing lists of Train and Test images have been saved to %s\n" % output_folder)


if __name__ == '__main__':
    # create_data_lists(train_folders=['/home/msiau/data/tmp/ibeltran/data/train2014/train2014_HR'],
    #                   test_folders=['/home/msiau/data/tmp/ibeltran/data/BSDS200/HR', '/home/msiau/data/tmp/ibeltran/data/urban100/HR',
    #                                 '/home/msiau/data/tmp/ibeltran/data/DIV2K/DIV2K_train_HR', '/home/msiau/data/tmp/ibeltran/data/General100/HR',
    #                                 '/home/msiau/data/tmp/ibeltran/data/manga109/HR', '/home/msiau/data/tmp/ibeltran/data/Set5/original',
    #                                '/home/msiau/data/tmp/ibeltran/data/Set14/original', '/home/msiau/data/tmp/ibeltran/data/T91/HR', ],
    #                   min_size=100,
    #                   output_folder='json')
    
    create_data_list_sat(data_folder="/home/msiau/data/tmp/ibeltran/sat/worldstrat/HR", 
                         min_size=100,
                         output_folder='json_sat')