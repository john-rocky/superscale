################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
from torchvision import transforms
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_imgvar
from utils.misc import create_npz_from_sample_folder
import pyiqa
import glob
import pickle
import numpy as np

MODEL_DEPTH = 24
assert MODEL_DEPTH in {16, 20, 24, 30}


# download checkpoint
vae_ckpt =  f'checkpoints/VQVAE.pth'
var_ckpt = f'checkpoints/VARSR_C2I.pth'
# build vae, var
#patch_nums = (1,2,3,4,6,9,13,18,24,32)
patch_nums = (1,2,3,4,6,9,12,16,20,24,28,32)
num_classes = 3830
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_imgvar(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=num_classes, depth=MODEL_DEPTH, shared_aln=False,
        fused_if_available=False, flash_if_available=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_local'], strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu')['trainer']['var_wo_ddp'], strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')
# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
cfg = 4.0 #@param {type:"slider", min:1, max:10, step:0.1}
more_smooth = False # True for more smooth output

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

base_name = "C2I/" 
os.makedirs(base_name, exist_ok=True)
# sample
batchsize = 4
iters = 1

classes = [13, 92, 190, 192, 260, 264, 292, 306, 335, 350, 428, 481, 516, 521, 534, 581, 592, 603, 633, 679, 697, 745, 763, 878, 918, 940, 949, 969, 983, 989, 1081, 1093, 1143, 1205, 1208, 1215, 1227, 1229, 1237, 1284, 1293, 1307, 1311, 1315, 1431, 1505, 1537, 1618, 1659, 1689, 1791, 1800, 0, 1871, 1891, 1892, 1900, 1922, 1979, 2019, 2135, 2197, 2284, 2296, 2375, 2450, 2513, 2520, 2536, 2541, 2564, 2566, 2592, 2646, 2711, 2746, 2757, 2783, 2789, 2818, 2923, 2928, 2964, 2998, 2999, 3114, 3175, 3238, 3245, 3251, 3266, 3302, 3338, 3344, 3366, 3435, 3474, 3615, 3708, 3713, 3798, 3799]


for class_labels in classes:
    B = batchsize
    for iter in range(iters):
        label_B = class_labels
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, text_hidden=None, negative_text= None, cfg=cfg, top_p=0.75, top_k=20, more_smooth=False)
                for i in range(B):
                    chw = recon_B3HW[i].permute(1, 2, 0).mul_(255).cpu().numpy()
                    chw = PImage.fromarray(chw.astype(np.uint8))
                    output_name = base_name + f'/class_{class_labels}_img_{iter*B+i}.png' 
                    chw.save(output_name)

