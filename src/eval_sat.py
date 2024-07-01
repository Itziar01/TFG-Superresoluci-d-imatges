import torch
import torch.nn.functional as F
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
from piq import LPIPS, DISTS, CLIPIQA
import sys
sys.path.append('/home/msiau/workspace/ibeltran/111/SRGAN')
from models import SRResNet, Generator, Discriminator
sys.path.append('/home/msiau/workspace/ibeltran/111/SRCNN')
from model import SRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = "/home/msiau/workspace/ibeltran/111/json_sat"
test_data_names = ["test_rgbn", "sen2venus", "sentinel2"] # 
image_types = ["rgbn", "rgbn_tensor", "rgbn_tensor"] # 

# Model checkpoints
srcnn_checkpoint = "/home/msiau/workspace/ibeltran/111/SRCNN/models/srcnn_sat_rgbn.pth.tar"
srresnet_checkpoint = "/home/msiau/workspace/ibeltran/111/SRGAN/models/checkpoint_srresnet_sat_rgbn.pth.tar"
srgan_checkpoint = "/home/msiau/workspace/ibeltran/111/SRGAN/models/checkpoint_srgan_sat_rgbn.pth.tar"

models = []
name_models = ["srcnn", "srresnet", "srgan"] # 
# Load model
srcnn_model = torch.load(srcnn_checkpoint)['model'].to(device)
models.append(srcnn_model.eval())

srresnet_model = torch.load(srresnet_checkpoint)['model'].to(device)
models.append(srresnet_model.eval())

srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
models.append(srgan_generator.eval())


# Initialize metric functions
lpips = LPIPS().to(device)
dists = DISTS().to(device)
clipiqa = CLIPIQA().to(device)



for model, name in zip(models, name_models):
    # Evaluate
    print('\n' + name)
    for test_data_name, type in zip(test_data_names, image_types):
        print("For %s:" % test_data_name)

        # if name == "srgan":
        #     lr_img_type = '[-1, 1]'
        # else:
        #     lr_img_type = 'imagenet-norm'

        # Custom dataloader
        test_dataset = SRDataset(data_folder,
                                split='test',
                                model_type=name,
                                image_type=type,
                                crop_size=0,
                                scaling_factor=4,
                                lr_img_type='[-1, 1]',
                                hr_img_type='[-1, 1]',
                                test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

        # Keep track of the metrics across batches
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()
        LPIPSs = AverageMeter()
        DISTSs = AverageMeter()
        LSTDs = AverageMeter()
        CLIPIQAs = AverageMeter()

        # Prohibit gradient computation explicitly to save memory
        with torch.no_grad():
            # Batches
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # Move to default device
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), [-1, 1]
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

                # Calculate PSNR and SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

                if sr_imgs.shape[1] > 3:
                    sr_imgs = sr_imgs[:,:-1,:,:]
                    hr_imgs = hr_imgs[:,:-1,:,:]
                # Calculate LPIPS
                lpips_value = lpips(sr_imgs, hr_imgs)
                LPIPSs.update(lpips_value.item(), lr_imgs.size(0))

                # Calculate DISTS
                dists_value = dists(sr_imgs, hr_imgs)
                DISTSs.update(dists_value.item(), lr_imgs.size(0))
                
                # Calculate CLIPIQA
                sr_imgs_255 = convert_image(sr_imgs, source='[-1, 1]', target='[0, 255]')
                clipiqa_value = clipiqa(sr_imgs_255)
                CLIPIQAs.update(clipiqa_value.item(), lr_imgs.size(0))


        # Print average metrics
        print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))
        print('LPIPS - {lpips.avg:.3f}'.format(lpips=LPIPSs))
        print('DISTS - {dists.avg:.3f}'.format(dists=DISTSs))
        print('CLIPIQA - {clipiqa.avg:.3f}\n'.format(clipiqa=CLIPIQAs))

    print("\n")


