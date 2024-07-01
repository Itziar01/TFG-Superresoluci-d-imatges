import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from piq import psnr, ssim, LPIPS, DISTS, CLIPIQA

from network_swinir import SwinIR as net
import utils_image as util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)
        
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['lpips'] = []
    test_results['dists'] = []
    test_results['clipiq'] = []
    psnr_val, ssim_val, lpips_val, dists_val, clipiq_val = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            try:
                output = test(img_lq, model, args, window_size)
            except:
                print("Algo en el test a fallado con " + path)
                continue
                
            output = output[..., :h_old * args.scale, :w_old * args.scale]
            
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        # cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)


        # evaluate psnr/ssim/lpips/dists/clipiq
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)
            psnr_val = util.calculate_psnr(output, img_gt, border=border)
            ssim_val = util.calculate_ssim(output, img_gt, border=border)
            # psnr_val = psnr(torch.tensor(output / 255.0).permute(2, 0, 1).unsqueeze(0), torch.tensor(img_gt / 255.0).permute(2, 0, 1).unsqueeze(0))
            # ssim_val = ssim(torch.tensor(output / 255.0).permute(2, 0, 1), torch.tensor(img_gt / 255.0).permute(2, 0, 1))
            lpips_val = LPIPS()(torch.tensor(output / 255.0).permute(2, 0, 1).unsqueeze(0), torch.tensor(img_gt / 255.0).permute(2, 0, 1).unsqueeze(0))
            dists_val = DISTS()(torch.tensor(output / 255.0).permute(2, 0, 1).unsqueeze(0), torch.tensor(img_gt / 255.0).permute(2, 0, 1).unsqueeze(0))
            clipiq_val = CLIPIQA()(torch.tensor(output).permute(2, 0, 1).unsqueeze(0))

            test_results['psnr'].append(psnr_val)
            test_results['ssim'].append(ssim_val)
            test_results['lpips'].append(lpips_val.item())
            test_results['dists'].append(dists_val.item())
            test_results['clipiq'].append(clipiq_val.item())

            # print(f'Testing {idx} {imgname} - PSNR: {psnr_val:.3f} dB; SSIM: {ssim_val:.3f}; LPIPS: {lpips_val.item():.3f}; DISTS: {dists_val.item():.3f}; CLIPIQA: {clipiq_val.item():.3f}')
        else:
            print(f'Testing {idx} {imgname}')

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
        ave_dists = sum(test_results['dists']) / len(test_results['dists'])
        ave_clipiq = sum(test_results['clipiq']) / len(test_results['clipiq'])

        print(f'\n{save_dir} \n-- Average PSNR: {ave_psnr:.3f} dB; Average SSIM: {ave_ssim:.3f}; Average LPIPS: {ave_lpips:.3f}; Average DISTS: {ave_dists:.3f}; Average CLIPIQA: {ave_clipiq:.3f}')

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
        
    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8
        
    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        print(f'{args.folder_lq}/{imgname}')
        img_lq = cv2.imread(f'{args.folder_lq}/{imgname}.png', cv2.IMREAD_COLOR).astype(
            np.float32) / 255.
        
    # 003 real-world image sr (load lq image only)
    elif args.task in ['real_sr']:
        # img_gt = None
        img_gt = cv2.imread(f'{args.folder_gt}/{imgname}.png', cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq = cv2.imread(f'{args.folder_lq}/{imgname}.png', cv2.IMREAD_COLOR).astype(
            np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == '__main__':
    main()
