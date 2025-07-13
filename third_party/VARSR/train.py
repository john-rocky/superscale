import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
import wandb
import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModel
from utils.lr_control import lr_wd_annealing

def from_pretrained_orig(var, state_dict):
    for k, v in var.state_dict().items():
        if '.cross_attn.' in k:
            if 'mat_q' in k:
                key = k.replace(".cross_attn", ".attn").replace("mat_q", "mat_qkv")
                state_dict[k] = state_dict[key][0:state_dict[key].shape[0]//3,:]
            elif 'mat_kv' in k:
                key = k.replace(".cross_attn", ".attn").replace("mat_kv", "mat_qkv")
                state_dict[k] = state_dict[key][state_dict[key].shape[0]//3:,:v.shape[1]]
            else:
                key = k.replace(".cross_attn", ".attn")
                state_dict[k] = state_dict[key]
        elif 'class_emb' in k:
            value = state_dict[k]
            if value.shape[0]>v.shape[0]:
                state_dict[k] = state_dict[k][:v.shape[0],:]
            elif value.shape[0] < v.shape[0]:
                state_dict[k] = torch.cat((state_dict[k][:3830,:], state_dict[k][:3830,:]), dim=0)
    for key, value in var.state_dict().items():
        if key in state_dict and state_dict[key].shape != value.shape:
            print(key)
            state_dict.pop(key)
    ret = var.load_state_dict(state_dict, strict=False)
    missing, unexpected = ret
    print(f'[VARTrainer.load_state_dict] missing:  {missing}')
    print(f'[VARTrainer.load_state_dict] unexpected:  {unexpected}')
    del state_dict

    return var

def build_everything(args: arg_util.Args):
    if args.wandb_flag:
        if dist.is_master():
            wandb_run = wandb.init(
                project="var",
                name=args.exp_name,
                reinit=True,
                config=args,
            )
        else:
            wandb_run = None
    else:
        wandb_run = None

    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # create tensorboard logger
    dist.barrier()
    
    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # build data
    vae_ckpt =  args.vae_model_path
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        num_classes, dataset_train, dataset_val = build_dataset(
            args.valid_data_path, final_reso=args.data_load_reso, tokenizer=None, null_text_ratio=0.3, original_image_ratio=0.0,
        )
        types = str((type(dataset_train).__name__, type(dataset_val).__name__))
        
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        del dataset_val
        
        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        
        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10
    
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VQVAE, build_var
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    
    vae_local, var_wo_ddp = build_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, controlnet_depth= args.depth,       # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums, control_patch_nums = args.patch_nums,
        num_classes=1+1, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    
    dist.barrier()
    vae_state = torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_local']
    vae_local.load_state_dict(vae_state, strict=True)

    var_state = torch.load(args.var_pretrain_path, map_location='cpu')["trainer"]["var_wo_ddp"]
    var_wo_ddp = from_pretrained_orig(var_wo_ddp, var_state)
    del vae_state, var_state

    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
    var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)


    print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')
    
    # build optimizer
    names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul', "diffloss"
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')

    var_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # build trainer
    trainer = VARTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var, text_encoder = None,
        clip_vision = None, exp_name = args.exp_name,
        var_opt=var_optim, label_smooth=args.ls, wandb_flag = args.wandb_flag,
    )

    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    del vae_local, var_wo_ddp, var, var_optim
    
    if start_it > 0:
        start_ep = 0
        while start_it >= iters_train:
            start_it = start_it - iters_train
            start_ep = start_ep + 1
    dist.barrier()
    return (
        trainer, start_ep, start_it,
        iters_train, ld_train, ld_val, wandb_run
    )

def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val, wandb_run,
    ) = build_everything(args)
    
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)

        # train
        step_cnt = 0
        me = misc.MetricLogger(delimiter='  ')
        me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
        me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
        header = f'[Ep]: [{ep:4d}/{args.ep}]'
        if ep == start_ep:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
        g_it, max_it = ep * iters_train, args.ep * iters_train
        if ep == start_ep:
            start_it = start_it
        else: 
            start_it = 0

        for it, (batch) in me.log_every(start_it, iters_train, ld_train, 300 if iters_train > 8000 else 5, header):
            g_it = ep * iters_train + it
            if it < start_it: continue
            if ep == start_ep and it == start_it: warnings.resetwarnings()
        
            inp = batch["pixel_values"].to(args.device, non_blocking=True)
            lr_inp = batch["conditioning_pixel_values"].to(args.device, non_blocking=True)
            label_B = batch["label_B"].to(args.device, non_blocking=True)        
            args.cur_it = f'{it+1}/{iters_train}'
            wp_it = args.wp * iters_train
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
            args.cur_lr, args.cur_wd = max_tlr, max_twd
        
            if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
                if g_it <= wp_it: prog_si = args.pg0
                elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
                else:
                    delta = len(args.patch_nums) - 1 - args.pg0
                    progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                    prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
            else:
                prog_si = -1
        
            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)
            
            grad_norm, scale_log2 = trainer.train_step(
                it=it, g_it=g_it, stepping=stepping, metric_lg=me,
                inp_B3HW=inp, lr_inp=lr_inp, label_B=label_B,
                text=None, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
                lr = args.cur_lr, wd =args.cur_wd 
            )
            me.update(tlr=max_tlr)
            if (g_it!=0 and g_it%1000==0) or ((it+1)%iters_train==0 and iters_train>1000):
                val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
                best_updated = best_val_loss_tail > val_loss_tail
                best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
                best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
                args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
                print(f' [*] [ep{ep}]  (val {tot})  Lm: {val_loss_mean:.4f}, Lt: {val_loss_tail:.4f}, Acc m&t: {val_acc_mean:.2f} {val_acc_tail:.2f},  Val cost: {cost:.2f}s')
                if dist.is_local_master():
                    local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                    local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                    print(f'[saving ckpt] ...', end='', flush=True)
                    torch.save({
                            'epoch':    ep+1,
                            'iter':     g_it,
                            'trainer':  trainer.state_dict(),
                            'args':     args.state_dict(),
                        }, local_out_ckpt)
                    if best_updated:
                        shutil.copy(local_out_ckpt, local_out_ckpt_best)
                    print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt_best}', flush=True, clean=True)
                    if args.wandb_flag and dist.is_master():
                        wandb.log({"val/L_mean": val_loss_mean, "val/L_tail": val_loss_tail, "val/acc_mean": val_acc_mean, "val/acc_tail": val_acc_tail})
                dist.barrier()
        
        me.synchronize_between_processes()
        stats = {k: meter.global_avg for k, meter in me.meters.items()}
        (sec, remain_time, finish_time) = me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time

        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        args.dump_log()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    if args.wandb_flag:
        if dist.is_master():
            wandb_run.finish()
    dist.barrier()


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
