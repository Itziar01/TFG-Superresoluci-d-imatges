import sys
import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model import SRCNN
sys.path.insert(1, '/home/msiau/workspace/ibeltran/111')
from utils import *
from datasets import SRDataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Data parameters
data_folder = '/home/msiau/workspace/ibeltran/111/json_sat'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
upscale_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Tensorboard
log_dir = 'srcnn_sat_rgbn'

# Model parameters
large_kernel = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel = 3  # kernel size of all convolutions in-betweens

# Learning parameters
checkpoint = False # load previous checkpoint?
name_checkpoint = "srcnn_sat_rgbn" # name to save checkpoint
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e9  # number of training iterations
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    '''
    Training
    '''
   
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is False:
        model = SRCNN(4, large_kernel, small_kernel, upscale_factor)
        # Initialize the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    else:
        checkpoint = torch.load(f"models/{name_checkpoint}.pth.tar")
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split='train',
                              crop_size=crop_size,
                              scaling_factor=upscale_factor,
                              model_type='srcnn',
                              image_type='rgbn',
                              lr_img_type='[-1, 1]',
                              hr_img_type='[-1, 1]')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    # tensorboard writer
    writer = SummaryWriter(f'runs/{log_dir}')

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              writer=writer)

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                    f"models/{name_checkpoint}.pth.tar")


def train(train_loader, model, criterion, optimizer, epoch, writer=None) -> None:
    '''
    One epoch's training.

    @train_loader (DataLoader) for training data
    @model (SRCNN)
    @criterion content loss function (Mean Squared-Error loss)
    @optimizer (SGD)
    @epoch (int) epoch number
    @writer (SummaryWriter) to save data in tensorboard
    '''
    
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (lr_image, hr_image) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_image = lr_image.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_image = hr_image.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_image = model(lr_image)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_image, hr_image)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip != None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_image.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))
            
    if writer is not None:
        writer.add_scalar("Training/Loss", losses.avg, epoch)
        writer.add_scalar("Training/BatchTime", batch_time.avg, epoch)

    del lr_image, hr_image, sr_image  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()