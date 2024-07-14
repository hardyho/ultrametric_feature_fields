import torch
from torch import nn
import torch.nn.functional as F
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
from graph import get_ultrametric
import scipy.sparse as sp 
import os
import random
import math
import time
os.environ["TORCH_HOME"] = "/tmp/torch/"

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")

import yaml


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'Sigmoid'
        if hparams.feature_directory is not None:
            assert hparams.feature_dim is not None, "set feature_dim for using feature field"
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act,
                         feature_out_dim=hparams.feature_dim)
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        

    def forward(self, batch, split, detach_geometry=False):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'black_bg': self.hparams.dataset_name == 'partnet',
                  'detach_geometry': detach_geometry}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.dataset_name == 'nerf' or self.hparams.dataset_name == 'partnet':
            kwargs['exp_step_factor'] = 0
            
        if self.hparams.depth_smooth and split=='train':
            kwargs['depth_smooth_samples_num'] = batch['depth_smooth_samples_num']
        else:
            batch['depth_smooth_samples_num'] = 0
        if split=='test':
            kwargs['render_feature'] = True

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        
        self.train_dataset = dataset(split=self.hparams.split,
                                     load_features=hparams.feature_directory is not None,
                                     feature_directory=hparams.feature_directory,
                                     load_seg=hparams.load_seg,
                                     load_depth_smooth=hparams.depth_smooth,
                                     hierarchical_sampling=hparams.hierarchical_sampling,
                                     num_seg_samples=hparams.num_seg_samples,
                                     neg_sample_ratio=hparams.neg_sample_ratio,
                                     **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy


        self.test_dataset = dataset(split='val', 
                                    load_seg=hparams.load_seg,
                                    num_seg_samples=hparams.num_seg_samples,
                                    neg_sample_ratio=hparams.neg_sample_ratio,
                                    rotate_test=hparams.rotate_test,
                                    render_train=hparams.render_train,
                                    render_train_subsample=hparams.render_train_subsample,
                                    **kwargs)
        self.test_dataset.batch_size = self.hparams.batch_size

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT'] and not n.startswith('val_lpips'):
                net_params += [p]
                print(n, p.shape, 'to be optimized')

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=8,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=4,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        torch.cuda.empty_cache()
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')
        
        results = self(batch, split='train')
        loss_d = self.loss(results, batch)

        if self.global_step % (2*self.update_interval) == 0:
            # regularization for cleaning
            loss_d['density_mean'] = self.model.sample_density(
                0.01*MAX_SAMPLES/3**0.5, warmup=self.global_step<self.warmup_steps).mean() * 1e-4

        # feature loss
        if 'feature' in results:
            loss_d['norm'] = (results['norm'] - 1) ** 2
            if self.hparams.ultrametric_weight != 0:
                positive_distance = get_ultrametric(results['feature'], 
                                                    results['rays_d'], 
                                                    batch['seg_positives'], batch['seg_masks'], visualize=(self.global_step%200==0),
                                                    pix_ids = batch['pix_idxs'], img_wh=batch['img_wh'])
                
                negative_distance = get_ultrametric(results['feature'], 
                                                    results['rays_d'], 
                                                    batch['seg_negatives'], batch['seg_masks'])
                
                N_mask, N_pairs = positive_distance.shape
                distances = torch.cat((positive_distance.reshape(N_mask * N_pairs, 1), negative_distance.reshape(N_mask * N_pairs, self.hparams.neg_sample_ratio)), dim=-1)
                
                temperature = 0.1
                distances = distances / temperature
                labels = torch.zeros((N_mask * N_pairs), dtype=torch.long).cuda()
                loss_d['loss_seg_ultrametric'] = F.cross_entropy(distances, labels) * self.hparams.ultrametric_weight
                
            if self.hparams.euclidean_weight != 0:
                positive_distance = (results['feature'][batch['seg_positives'][..., 0]] *
                                    results['feature'][batch['seg_positives'][..., 1]]).sum(-1) # 
                negative_distance = (results['feature'][batch['seg_negatives'][..., 0]] *
                                    results['feature'][batch['seg_negatives'][..., 1]]).sum(-1)  # 
                
                N_mask, N_pairs = positive_distance.shape
                distances = torch.cat((positive_distance.reshape(N_mask * N_pairs, 1), negative_distance.reshape(N_mask * N_pairs, self.hparams.neg_sample_ratio)), dim=-1)
                
                temperature = 0.1
                distances = distances / temperature
                labels = torch.zeros((N_mask * N_pairs), dtype=torch.long).cuda()
                loss_d['loss_seg_euclidean'] = F.cross_entropy(distances, labels) * self.hparams.euclidean_weight
        
            
        # Add the depth smoothing loss after 5 epochs 
        if self.hparams.depth_smooth and self.current_epoch > 2:
            loss_d['loss_depth_smooth'] = loss_d['norm'] * 0
            EPS = 1e-2
            depth_pairs = results['depth_sup'][:batch['depth_smooth_samples_num']].reshape(-1, 4)
            # Only supervising high-opacity pixels to avoid noise
            mask = (results['opacity_sup'][:batch['depth_smooth_samples_num']].reshape(-1, 4) > 0.9).all(dim=-1)
            depth_pairs = depth_pairs[mask]
            
            if mask.any():
                loss_d['loss_depth_smooth'] = torch.clip(torch.abs(depth_pairs[:, 0] - 3 * depth_pairs[:, 1] + 3 * depth_pairs[:, 2] - depth_pairs[:, 3]) / 
                                                    (depth_pairs.max(dim=1)[0].detach() + EPS) ** 3 - 0.1, 0).mean() 
           

        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v.mean())
        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if (not self.hparams.no_save_test):
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.current_epoch}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):       
        logs = {}
        w, h = self.train_dataset.img_wh
        with torch.no_grad():
            results = self(batch, split='test')
            
        # compute each metric per image
        rgb_gt = batch['rgb']
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()
        
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                        torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        # Get per-frame 2D segmentation result with the watershed transform algorithm
        def get_segmentation(feat, distance, num_keep, h, w):
            rows = []
            cols = []

            # Building the graph
            rows.append(torch.arange(0, (h-1) * w))
            cols.append(torch.arange(w, h*w))
            rows.append(torch.tensor([i for i in range(h*w) if i % w != (w-1)]))
            cols.append(torch.tensor([(i + 1) for i in range(h*w) if i % w != (w-1)]))
            rows = torch.cat(rows)
            cols = torch.cat(cols)
            data = (((feat[rows] - feat[cols]) ** 2).sum(-1) + 1e-8).sqrt()
            rows = rows[data < distance]
            cols = cols[data < distance]
            data = data[data < distance]

            graph = sp.csr_matrix((data.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())), shape=(len(feat), len(feat)))
            num_components, segmentation = sp.csgraph.connected_components(graph, connection='weak')
            
            count = np.histogram(segmentation, bins=[i for i in range(segmentation.max() + 2)])[0]
            
            # Reassign the id according to the size of each mask
            segmentation = count.argsort()[::-1].argsort()[segmentation]
            segmentation[segmentation > num_keep] = num_keep # keep the num_keep largest masks
            
            return segmentation
        
        if self.hparams.run_seg_inference:
            colors = np.random.randint(0, 256, (self.hparams.num_seg_test + 1, 3))
            colors[-1] = 255
        
        if (not self.hparams.no_save_test): # save test image to disk
            idx = batch['img_idxs'] 
            segmentations = []
            distances = [(i+1) * 0.01 for i in range(100)]

            if self.current_epoch == self.hparams.num_epochs - 1 and self.hparams.run_seg_inference and idx > self.test_dataset.num_training_frames:
                for distance in distances:
                    segmentation = get_segmentation(results['feature'], distance, self.hparams.num_seg_test, h, w)
                    segmentations.append(segmentation)
                    
                    segmentation_map = rearrange(segmentation, '(h w) -> h w', h=h)
                    segmentation_map = (colors[segmentation_map]).astype(np.uint8)
                    seg_dir = os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}_s')
                    os.makedirs(seg_dir, exist_ok=True)
                    imageio.imsave(os.path.join(seg_dir, f'{distance:.3f}.png'), segmentation_map)

                if self.hparams.rotate_test:
                    # There's no ground truth for rotate_test mode
                    accuracy = torch.tensor(0.)
                else:
                    segmentations = torch.stack([torch.tensor(s).cuda() for s in segmentations])
                    accuracy, indices = (((segmentations[:, batch['seg_positives'][..., 0]] == segmentations[:, batch['seg_positives'][..., 1]]).sum(-1) / batch['seg_positives'].shape[1] + 
                                        (segmentations[:, batch['seg_negatives'][..., 0]] != segmentations[:, batch['seg_negatives'][..., 1]]).sum(-1) / batch['seg_negatives'].shape[1])/2).max(dim=0)
                    accuracy = accuracy.mean()
            
            else:
                accuracy = torch.tensor(0.)
                
            logs['acc'] = accuracy
                
            # visualize PCA feature
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
                if not hasattr(self, 'proj_V'):
                    U, S, V = torch.pca_lowrank(
                        (results['feature']).float(),
                        niter=5)
                    self.proj_V = V[:, :3].float()
                    lowrank = torch.matmul(results['feature'].float(), self.proj_V)
                    self.lowrank_sub = lowrank.min(0, keepdim=True)[0]
                    self.lowrank_div = lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0]
                else:
                    lowrank = torch.matmul(results['feature'].float(), self.proj_V)
                lowrank = ((lowrank - lowrank.min(0, keepdim=True)[0]) / (lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0])).clip(0, 1)
                visfeat = rearrange(lowrank.cpu().numpy(), '(h w) c -> h w c', h=h)
                visfeat = (visfeat*255).astype(np.uint8)
                if self.current_epoch % 5 == 0 or self.current_epoch == self.hparams.num_epochs - 1:  
                    if idx < self.test_dataset.num_training_frames:
                        imageio.imsave(os.path.join(self.val_dir, f'train_{idx:03d}_f.png'), visfeat)
                    else:
                        imageio.imsave(os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}_f.png'), visfeat)

            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            
            if self.current_epoch % 1 == 0 or self.current_epoch == self.hparams.num_epochs - 1:  
                depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                if idx < self.test_dataset.num_training_frames:
                    imageio.imsave(os.path.join(self.val_dir, f'train_{idx:03d}.png'), rgb_pred)
                    imageio.imsave(os.path.join(self.val_dir, f'train_{idx:03d}_d.png'), depth)
                else:
                    imageio.imsave(os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}.png'), rgb_pred)
                    imageio.imsave(os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}_d.png'), depth)
                
            if self.current_epoch == self.hparams.num_epochs - 1:
                if idx < self.test_dataset.num_training_frames:
                    np.save(os.path.join(self.val_dir, f'train_{idx:03d}_d.npy'), results['depth'].cpu().numpy())
                    np.save(os.path.join(self.val_dir, f'train_{idx:03d}_p.npy'), batch['pose'].cpu().numpy())
                    torch.save(results['feature'].detach().cpu(), os.path.join(self.val_dir, f'train_{idx:03d}_f.pth'))
                else:
                    np.save(os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}_d.npy'), results['depth'].cpu().numpy())
                    np.save(os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}_p.npy'), batch['pose'].cpu().numpy())
                    torch.save(results['feature'].detach().cpu(), os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}_f.pth'))
                
                np.save(os.path.join(self.val_dir, f'intrinsic.npy'), batch['intrinsic'].cpu().numpy())
                
                if 'surface' in results:
                    surface = rearrange(results['surface'].cpu().numpy(), '(h w) c -> h w c', h=h)
                    if idx < self.test_dataset.num_training_frames:
                        np.save(os.path.join(self.val_dir, f'train_{idx:03d}_surface.npy'), surface)
                    else:
                        np.save(os.path.join(self.val_dir, f'{idx - self.test_dataset.num_training_frames:03d}_surface.npy'), surface)
                    print('Surface saved.')

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)
            
        if 'acc' in outputs[0]:
            accs = torch.stack([x['acc'] for x in outputs])
            mean_acc = all_gather_ddp_if_available(accs).mean()
            self.log('test/acc', mean_acc)
            print("Accuracy:", mean_acc)

        if 'beta' in outputs[0]:
            betas = torch.stack([x['beta'] for x in outputs])
            mean_beta = all_gather_ddp_if_available(betas).mean()
            self.log('test/beta', mean_beta)
            
        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=False,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = WandbLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=1,
                      callbacks=callbacks,
                      logger=logger,
                      log_every_n_steps=50,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      accumulate_grad_batches=hparams.accumulate_grad_batches,
                      # amp_backend="apex",
                      # amp_level="O1",
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')
        print(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')
