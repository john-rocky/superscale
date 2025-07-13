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

logger = get_logger(__name__, log_level="INFO")

def main(args: arg_util.Args):
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
    val_set = []
    for folder in folders:
        dataset_val = TestDataset("testset/" + folder, image_size=args.data_load_reso, tokenizer=None, resize_bak=True)
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        val_set.append(ld_val)

    for ld_val in val_set:
        for batch in ld_val:
            lr_inp = batch["conditioning_pixel_values"].to(args.device, non_blocking=True)
            label_B = batch["label_B"].to(args.device, non_blocking=True)
            B = lr_inp.shape[0]

            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                    start_time = time.time()
                    recon_B3HW = var.autoregressive_infer_cfg(B=B, cfg=6.0, top_k=1, top_p=0.75,
                                                        text_hidden=None, lr_inp=lr_inp, negative_text=None, label_B=label_B, lr_inp_scale = None,
                                                        more_smooth=False)
                    recon_B3HW = numpy_to_pil(pt_to_numpy(recon_B3HW))

            for idx in range(B):
                image = recon_B3HW[idx]
                if True: 
                    validation_image = Image.open(batch['path'][idx].replace("/HR","/LR")).convert("RGB")
                    validation_image = validation_image.resize((512, 512))
                    image = adain_color_fix(image, validation_image)

                folder_path, ext_path = os.path.split(batch['path'][idx])
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

    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    fid_metric = pyiqa.create_metric('fid', device=device)
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
    clipiqa_iqa_metric = pyiqa.create_metric('clipiqa', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    dists_iqa_metric = pyiqa.create_metric('dists', device=device)
    niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)

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
        gt_img_paths.extend(sorted(glob.glob(f'{dir}/{folder}/HR/*.JPEG'))[:])
        gt_img_paths.extend(sorted(glob.glob(f'{dir}/{folder}/HR/*.png'))[:])
        real_image_folder = dir + "/" + folder + "/HR"
        generated_image_folder = real_image_folder.replace("/HR", "/VARPrediction")

        for gt_img_path in gt_img_paths:
            GT_image = img_preproc(Image.open(gt_img_path).convert('RGB'))
            prediction_img_path = gt_img_path.replace("/HR/", "/VARPrediction/")
            VARPrediction_img = img_preproc(Image.open(prediction_img_path).convert('RGB'))

            img1 = rgb2ycbcr_pt(img2tensor(io.imread(gt_img_path)),  y_only=True).to(torch.float64)
            img2 = rgb2ycbcr_pt(img2tensor(io.imread(prediction_img_path)),  y_only=True).to(torch.float64)
            img1 = torch.squeeze(img1)
            img2 = torch.squeeze(img2)
            
            ssim_folder.append(ssim_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            psnr_folder.append(psnr_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            lpips_iqa.append(lpips_iqa_metric(prediction_img_path, gt_img_path))
            clip_iqa.append(clipiqa_iqa_metric(prediction_img_path))
            musiq_iqa.append(musiq_iqa_metric(prediction_img_path))
            maniqa_iqa.append(maniqa_metric(prediction_img_path))
            dists_score.append(dists_iqa_metric(prediction_img_path, gt_img_path))
            niqe_score.append(niqe_iqa_metric(prediction_img_path))

        m_psnr = sum(psnr_folder) / len(psnr_folder)
        m_ssim = sum(ssim_folder) / len(ssim_folder)
        print(f"PSNR = {m_psnr}")
        print(f"SSIM = {m_ssim}")
        m_lpips = sum(lpips_iqa)/len(lpips_iqa)
        print(f"LPIPS = {m_lpips.item()}")
        m_dists = sum(dists_score)/len(dists_score)
        print(f"DISTS = {m_dists}")
        m_niqe = sum(niqe_score)/len(niqe_score)
        print(f"NIQE = {m_niqe}")
        clipiqa = sum(clip_iqa)/len(clip_iqa)
        print(f"CLIP-IQA = {clipiqa.item()}")
        musiq = sum(musiq_iqa)/len(musiq_iqa)
        print(f"MUSIQ = {musiq.item()}")
        maniqa = sum(maniqa_iqa)/len(maniqa_iqa)
        print(f"MANIQA = {maniqa.item()}")
        fid_value = fid_metric(real_image_folder, generated_image_folder)
        print(f"FID = {fid_value}")

    return (m_psnr.item(), m_ssim.item(), m_lpips.item(), m_dists.item(), clipiqa.item(), musiq.item(), maniqa.item())



if __name__ == "__main__":
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main(args)
    results = metrics()
