from PIL import Image
import random
import torchvision.transforms.functional as FT
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

def convert_image(img, source, target):
    '''
    Convert an image from a source format to a target format.

    @img (tensor or PIL image) image
    @source (str) source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    @target (str) target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
    :return: converted image
    '''
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    return img


class ImageTransforms(object):
    '''
    Image transformation pipeline.
    '''

    def __init__(self, model_type, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
        '''
        @model_type (str) name of the moodel
        @split (str) one of 'train' or 'test'
        @crop_size (int) crop size of HR images
        @scaling_factor (int) LR images will be downsampled from the HR images by this factor
        @lr_img_type (str) the target format for the LR image; see convert_image() above for available formats
        @hr_img_type (str) the target format for the HR image; see convert_image() above for available formats
        '''
        self.model_type = model_type.lower()
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        '''
        @img (PIL image) from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        '''

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        if self.model_type == 'srcnn':
            lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)
            lr_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)
            
            # Sanity check
            assert hr_img.width == lr_img.width and hr_img.height == lr_img.height
        else:
            lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                                Image.BICUBIC)

            # Sanity check
            assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


class AverageMeter(object):
    '''
    Keeps track of most recent, average, sum, and count of a metric.
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    '''
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    @optimizer optimizer with the gradients to be clipped
    @grad_clip clip value
    '''
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    '''
    Save model checkpoint.

    @state checkpoint contents
    '''

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    '''
    Shrinks learning rate by a specified factor.

    @optimizer optimizer whose learning rate must be shrunk.
    @shrink_factor factor in interval (0, 1) to multiply learning rate with.
    '''

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))