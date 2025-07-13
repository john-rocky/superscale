import os
import sys
import glob
import argparse
import numpy as np
import yaml
from PIL import Image
import torch.nn.functional as F
import safetensors.torch
import time
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
import dist
import torch
from torchvision import transforms
import torch.utils.checkpoint
from utils import arg_util, misc
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPVisionModel, CLIPTokenizer, CLIPImageProcessor
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from dataloader.testdataset import TestDataset
import math
from torch.utils.data import DataLoader
from torchvision import transforms
import pyiqa
from skimage import io
from models import VAR_RoPE, VQVAE, build_var

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img
def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor
def numpy_to_pil(images: np.ndarray):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images
def gaussian_weights(tile_width, tile_height, nbatches):
    """Generates a gaussian mask of weights for tile contributions"""
    from numpy import pi, exp, sqrt
    import numpy as np

    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=dist.get_device()), (nbatches, 32, 1, 1))



logger = get_logger(__name__, log_level="INFO")

def main(args: arg_util.Args):
    device = args.device
    vae_ckpt =  args.vae_model_path
    var_ckpt = args.var_test_path
    args.depth = 24

    vae, var = build_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, controlnet_depth=args.depth,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums, control_patch_nums =args.patch_nums,
        num_classes=1 + 1, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_local'], strict=True)
    model_state = torch.load(var_ckpt, map_location='cpu')
    var.load_state_dict(model_state['trainer']['var_wo_ddp'], strict=True)
    vae.eval(), var.eval()


    img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])


    image_names = []
    folders = os.listdir("testset/")
    for folder in folders:
        image_names.extend(sorted(glob.glob(f'testset/{folder}/LR/*.png'))[:])
    
    acc_mean = 0.0
    rscale = 4
    for image_name in image_names[:]:
        lr_inp = Image.open(image_name).convert("RGB")
        ori_h = lr_inp.size[0]*rscale
        ori_w = lr_inp.size[1]*rscale

        lr_inp = lr_inp.resize((max(math.ceil(lr_inp.size[0]/16)*16*rscale,512), max(math.ceil(lr_inp.size[1]/16)*16*rscale, 512))) 
        lr_inp = img_preproc(lr_inp).unsqueeze(0) * 2.0 - 1.0
        lr_inp = lr_inp.to(dist.get_device(), non_blocking=True)

        label_B = torch.zeros(1).to(dist.get_device(),  dtype = int, non_blocking=True)
        B = lr_inp.shape[0]
        h, w = math.ceil(lr_inp.shape[2]/16), math.ceil(lr_inp.shape[3]/16)
        tile_size = 32
        tile_overlap = 8
        tile_weights = gaussian_weights(32, 32, 1)

        grid_rows = 0
        cur_x = 0
        while cur_x < h:
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < w:
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1
        recon_pred = []
        start_time = time.time()
        for row in range(grid_rows):
            input_lr = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = h - tile_size
                if col == grid_cols-1:
                    ofs_y = w - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size
                # input tile dimensions
                input_tile = lr_inp[:, :, input_start_x*16:input_end_x*16, input_start_y*16:input_end_y*16]
                input_lr.append(input_tile)

                if col == grid_cols-1:
                    if len(input_lr)>1:
                        lr4var = torch.cat(input_lr, dim=0)
                    else:
                        lr4var = input_lr[0]

                    print(lr4var.shape)
                    with torch.inference_mode():
                        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                            recon_B3HW = var.autoregressive_infer_cfg(B=grid_cols, cfg=7.0, top_k=1, top_p=0.75,
                                                                text_hidden=None, lr_inp=lr4var, negative_text=None, label_B=label_B.repeat(grid_cols), lr_inp_scale = None, tile_flag=True,
                                                                more_smooth=False)
                            recon_pred.append(recon_B3HW)

        preds = torch.zeros((B, 32, h, w), device=lr_inp.device)
        contributors = torch.zeros((B, 32, h, w), device=lr_inp.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = h - tile_size
                if col == grid_cols-1:
                    ofs_y = w - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                preds[:, :, input_start_x:input_end_x, input_start_y:input_end_y] += recon_pred[row][col].unsqueeze(0) * tile_weights
                contributors[:, :, input_start_x:input_end_x, input_start_y:input_end_y] += tile_weights
        # Average overlapping areas with more than 1 contributor
        preds /= contributors
        with torch.no_grad():
            recon_B3HW = vae.fhat_to_img(preds).add_(1).mul_(0.5)
        recon_B3HW = numpy_to_pil(pt_to_numpy(recon_B3HW))
        end_time = time.time()
        duration = end_time - start_time
        print(duration)

        for idx in range(B):
            image = recon_B3HW[idx].resize((ori_h, ori_w))
            if True: 
                validation_image = Image.open(image_name).convert("RGB")
                validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
                image = adain_color_fix(image, validation_image)

            folder_path, ext_path = os.path.split(image_name)
            output_name = folder_path.replace("/LR", "/VARPrediction/").replace("/HR", "/VARPrediction/")
            os.makedirs(output_name, exist_ok=True)
            image.save(os.path.join(output_name, ext_path))

    return True


def metrics():
    dir = "testset/"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(pyiqa.list_models())
    folders = os.listdir("testset/")
    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    clipiqa_iqa_metric = pyiqa.create_metric('clipiqa', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)

    print(prediction_dir)
    for folder in folders:
        print(folder)
        gt_img_paths = []

        psnr_folder = []
        ssim_folder = []
        lpips_score = []
        dists_score = []
        niqe_score = []
        lpips_iqa = []
        musiq_iqa = []
        maniqa_iqa = []
        clip_iqa = []
        gt_img_paths.extend(sorted(glob.glob(f'{dir}/{folder}/LR/*.JPEG'))[:])
        gt_img_paths.extend(sorted(glob.glob(f'{dir}/{folder}/LR/*.png'))[:])
        real_image_folder = dir + "/" + folder + "/LR"
        generated_image_folder = real_image_folder.replace("/LR", "/VARPrediction/" + prediction_dir)

        for gt_img_path in gt_img_paths:
            prediction_img_path = gt_img_path.replace("/HR/", "/VARPrediction/")
            VARPrediction_img = img_preproc(Image.open(prediction_img_path).convert('RGB'))

            clip_iqa.append(clipiqa_iqa_metric(prediction_img_path))
            musiq_iqa.append(musiq_iqa_metric(prediction_img_path))
            maniqa_iqa.append(maniqa_metric(prediction_img_path))

        clipiqa = sum(clip_iqa)/len(clip_iqa)
        print(f"CLIP-IQA = {clipiqa.item()}")
        musiq = sum(musiq_iqa)/len(musiq_iqa)
        print(f"MUSIQ = {musiq.item()}")
        maniqa = sum(maniqa_iqa)/len(maniqa_iqa)
        print(f"MANIQA = {maniqa.item()}")




if __name__ == "__main__":
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main(args)
    results = metrics()
