import os
import time
from pathlib import Path

import imageio
#import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from einops import rearrange
from models.rendering import render
from models.networks import NGP
from opt import get_opts
from tqdm import tqdm
# from train import NeRFSystem, depth2img
from utils import load_ckpt

import clip
import yaml
import os
from clip_utils import CLIPEditor
#os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch/"
os.environ["TORCH_HOME"] = "/tmp/torch/"

if __name__ == "__main__":
    hparams = get_opts()
    os.makedirs(hparams.render_dir, exist_ok=True)

    rgb_act = 'Sigmoid'
    model = NGP(scale=hparams.scale, rgb_act=rgb_act,
                         feature_out_dim=hparams.feature_dim).cuda()
    load_ckpt(model, hparams.ckpt_path, prefixes_to_ignore=["density_grid", "grid_coords"])
    
    
    kwargs = {'root_dir': hparams.root_dir,
                'downsample': hparams.downsample}
    # dataset
    dataset = dataset_dict[hparams.dataset_name](split='val', 
            load_seg=hparams.load_seg,
            num_seg_samples=hparams.num_seg_samples,
            neg_sample_ratio=hparams.neg_sample_ratio,
            rotate_test=hparams.rotate_test,
            render_train=hparams.render_train,
            render_train_subsample=hparams.render_train_subsample,
            **kwargs)

    # start
    directions = dataset.directions.cuda()
    
    for img_idx in tqdm(range(len(dataset))):
        poses = dataset[img_idx]["pose"].cuda()
        rays_o, rays_d = get_rays(directions, poses)
        kwargs = {'test_time': True,
            'random_bg': hparams.random_bg,
            'black_bg': hparams.dataset_name == 'partnet',
            'detach_geometry': True}
        if hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if hparams.dataset_name == 'nerf' or hparams.dataset_name == 'partnet':
            kwargs['exp_step_factor'] = 0
            
        kwargs['render_feature'] = True

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            results = render(model, rays_o, rays_d, **kwargs)

        w, h = dataset.img_wh
        image = results["rgb"].reshape(h, w, 3)
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        if img_idx < dataset.num_training_frames:
            imageio.imsave(os.path.join(hparams.render_dir, f"train_{img_idx:03d}.png"), image)  # TODO: cv2
        else:
            imageio.imsave(os.path.join(hparams.render_dir, f"{img_idx - dataset.num_training_frames:03d}.png"), image)  # TODO: cv2
        
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            U, S, V = torch.pca_lowrank(
                            (results['feature']).float(),
                            niter=5)
            proj_V = V[:, :3].float()
            lowrank = torch.matmul(results['feature'].float(), proj_V)
            lowrank_sub = lowrank.min(0, keepdim=True)[0]
            lowrank_div = lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0]
        
            lowrank = ((lowrank - lowrank.min(0, keepdim=True)[0]) / (lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0])).clip(0, 1)
            visfeat = rearrange(lowrank.cpu().numpy(), '(h w) c -> h w c', h=h)
            visfeat = (visfeat*255).astype(np.uint8)
            if img_idx < dataset.num_training_frames:
                imageio.imsave(os.path.join(hparams.render_dir, f"train_{img_idx:03d}_f.png"), visfeat)
                torch.save(results['feature'], os.path.join(hparams.render_dir, f"train_{img_idx:03d}_f.pth"))
            else:
                imageio.imsave(os.path.join(hparams.render_dir, f"{img_idx - dataset.num_training_frames:03d}_f.png"), visfeat)
                torch.save(results['feature'], os.path.join(hparams.render_dir, f"{img_idx - dataset.num_training_frames:03d}_f.pth"))
            

            if img_idx < dataset.num_training_frames:
                np.save(os.path.join(hparams.render_dir, f'train_{img_idx:03d}_d.npy'), results['depth'].cpu().numpy())
                np.save(os.path.join(hparams.render_dir, f'train_{img_idx:03d}_p.npy'), dataset[img_idx]['pose'].cpu().numpy())
                torch.save(results['feature'].detach().cpu(), os.path.join(hparams.render_dir, f'train_{img_idx:03d}_f.pth'))
            else:
                np.save(os.path.join(hparams.render_dir, f'{img_idx - dataset.num_training_frames:03d}_d.npy'), results['depth'].cpu().numpy())
                np.save(os.path.join(hparams.render_dir, f'{img_idx - dataset.num_training_frames:03d}_p.npy'), dataset[img_idx]['pose'].cpu().numpy())
                torch.save(results['feature'].detach().cpu(), os.path.join(hparams.render_dir, f'{img_idx - dataset.num_training_frames:03d}_f.pth'))
            
            np.save(os.path.join(hparams.render_dir, f'intrinsic.npy'), dataset[img_idx]['intrinsic'].cpu().numpy())
            
            if 'surface' in results:
                surface = rearrange(results['surface'].cpu().numpy(), '(h w) c -> h w c', h=h)
                if img_idx < dataset.num_training_frames:
                    np.save(os.path.join(hparams.render_dir, f'train_{img_idx:03d}_surface.npy'), surface)
                else:
                    np.save(os.path.join(hparams.render_dir, f'{img_idx - dataset.num_training_frames:03d}_surface.npy'), surface)
                print('Surface saved.')
