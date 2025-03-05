import numpy as np
import os
import time
from pprint import pprint
import argparse
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage import restoration
from data import create_dataloader
from models import deepdeblur
from utils.utils import save_image, AverageMeter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# settings
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id to use')
parser.add_argument('--n_cpu', type=int, default=5,
                    help='cpus for data processing')

# data parameters
parser.add_argument('--dataset', type=str, default='GOPRO_Large',
                    help='GOPRO | GOPRO_Large')
parser.add_argument('--blur_type', type=str, default='lin',
                    help='gamma | lin')

parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='path to the checkpoint to load')
parser.add_argument('--save_dir', type=str, default=None,
                    help='directory name to save results (default: YY-MM-DD-HHMMSS)')

parser.add_argument('--only_ai', default=False, action=argparse.BooleanOptionalAction)


def main():

    args = parser.parse_args()
    pprint(vars(args))

    args.dataset = os.path.join('./datasets', args.dataset)

    if args.save_dir is None:
        args.save_dir = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
    args.save_dir = os.path.join('./results', args.save_dir)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs('results', exist_ok=True)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        raise ValueError('save dir exists!')

    test_loader = create_dataloader(args, phase='test')

    model = deepdeblur.DeepDeblur_scale3()
    model.to(device)

    if args.checkpoint_path is None:
        print('no checkpoint is specified.. load pre-trained model')
        args.checkpoint_path = './checkpoints/pretrained/430.pth'
    pretrained_dict = torch.load(args.checkpoint_path,map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    test(test_loader, model, device, args)
    if args.only_ai is False:
        test_loader = create_dataloader(args, phase='wiener')
        test_wiener(test_loader, args)


k = 4
psf = np.ones((k,k)) / (k*k)

def test(test_loader, model, device, args):
    test_psnr_ai = AverageMeter('PSNR (AI)')
    test_ssim_ai = AverageMeter('SSIM (AI)')

    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader, 1):
            blur1, blur2, blur3 = input
            sharp1, sharp2, sharp3 = label
            blur1, blur2, blur3 = blur1.to(device), blur2.to(device), blur3.to(device)
            sharp1, sharp2, sharp3 = sharp1.to(device), sharp2.to(device), sharp3.to(device)

            pred1, _, _ = model(blur1, blur2, blur3)

            sharp1 = sharp1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)
            pred1 = pred1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)
            blur1 = blur1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)

            sharp1 += 0.5
            pred1 += 0.5
            blur1 += 0.5

            # Calculate PSNR and SSIM for the AI model output
            psnr_ai = peak_signal_noise_ratio(sharp1, pred1, data_range=1.)
            ssim_ai = structural_similarity(sharp1, pred1, multichannel=True, win_size=3, gaussian_weights=True, use_sample_covariance=False, data_range=1)


            # Update meters
            test_psnr_ai.update(psnr_ai)
            test_ssim_ai.update(ssim_ai)

            # Print metrics (f-string is now properly used)
            print(f'{i}/{len(test_loader)} | '
                  f'PSNR (AI) {psnr_ai:.2f} | SSIM (AI) {ssim_ai:.4f} | ')

            # Save images
            save_image(sharp1, os.path.join(args.save_dir, f'{i}_sharp.png'))
            save_image(blur1, os.path.join(args.save_dir, f'{i}_blur.png'))
            save_image(pred1, os.path.join(args.save_dir, f'{i}_pred.png'))

    print(f'>> Avg PSNR (AI): {test_psnr_ai.avg:.2f}, Avg SSIM (AI): {test_ssim_ai.avg:.4f}')

def deblur_wiener(img_blur):

    # Apply Wiener filter to each color channel
    imgRestor_r = restoration.wiener(img_blur[:, :, 0], psf, balance=0.50)
    imgRestor_g = restoration.wiener(img_blur[:, :, 1], psf, balance=0.50)
    imgRestor_b = restoration.wiener(img_blur[:, :, 2], psf, balance=0.50)

    imgRestor_rgb = np.stack((imgRestor_r, imgRestor_g, imgRestor_b), axis=-1)

    imgRestor_rgb = np.clip(imgRestor_rgb, 0, 1)

    # Convert the image to uint8 for display
    # imgRestor_rgb = (imgRestor_rgb * 255).astype(np.uint8)

    return imgRestor_rgb



def test_wiener(test_loader, args):
    test_psnr_wiener = AverageMeter('PSNR (Wiener)')
    test_ssim_wiener = AverageMeter('SSIM (Wiener)')

    for i, (input, label) in enumerate(test_loader, 1):
        blur1, blur2, blur3 = input
        sharp1, sharp2, sharp3 = label

        # Convert tensors to numpy arrays and transpose them to (H, W, C)
        sharp1 = sharp1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)
        blur1 = blur1.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0)

        sharp1 = np.clip(sharp1 + 0.5, 0, 1)  # If the image was normalized to [-0.5, 0.5]
        blur1 = np.clip(blur1 + 0.5, 0, 1)

        # Apply the Wiener deblur
        blur1_wiener = deblur_wiener(blur1)

        # Calculate PSNR and SSIM
        psnr_wiener = peak_signal_noise_ratio(sharp1, blur1_wiener, data_range=1)  # Convert to [0, 255]
        ssim_wiener = structural_similarity(sharp1, blur1_wiener, gaussian_weights = True,win_size=11,  data_range=1)

        # Update the average PSNR and SSIM
        test_psnr_wiener.update(psnr_wiener)
        test_ssim_wiener.update(ssim_wiener)

        print(f'{i}/{len(test_loader)} | '
              f'PSNR (Wiener) {psnr_wiener:.2f} | SSIM (Wiener) {ssim_wiener:.4f}')

        # Save image
        save_image(blur1_wiener, os.path.join(args.save_dir, f'{i}_wiener_deblur.png'))

    print(f'>> Avg PSNR (Wiener): {test_psnr_wiener.avg:.2f}, Avg SSIM (Wiener): {test_ssim_wiener.avg:.4f}')


if __name__ == '__main__':
    main()