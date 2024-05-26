from torch.utils.data import Dataset
import numpy as np
import torch
from .ray_utils import get_rays
import random

# filter all masks with less than 5 sampled pixels
FILTER_THRES = 5

class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0, len_per_epoch=1000):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.len_per_epoch = len_per_epoch
        self.patch_size = None  # 128
        self.patch_coverage = 0.9

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return self.len_per_epoch
        return len(self.poses)

    def sample_patch(self, h, w):
        skip = int((min(h, w) * self.patch_coverage) / self.patch_size)
        patch_w_skip = self.patch_size * skip
        patch_h_skip = self.patch_size * skip

        left = torch.randint(0, w - patch_w_skip - 1, (1,))[0]
        left_to_right = torch.arange(left, left + patch_w_skip, skip)
        top = torch.randint(0, h - patch_h_skip - 1, (1,))[0]
        top_to_bottom = torch.arange(top, top + patch_h_skip, skip)

        index_hw = (top_to_bottom * w)[:, None] + left_to_right[None, :]
        return index_hw.reshape(-1)

    def get_seg_samples(self, masks, seg_hierarchy):
        def _get_hierarchical_samples(mask, masks, mask_idx, last_negative_sample=None, last_negative_idx=None):
            # If it's a child
            if mask_idx in seg_hierarchy[:, 0]:
                # All masks that contain the current one
                ancestors = seg_hierarchy[(seg_hierarchy[:, 0] == mask_idx)][:, 1].unique()
                
                # Masks that are not a parent of another ancestor
                parents = ancestors[~torch.isin(ancestors, seg_hierarchy[torch.isin(seg_hierarchy[:, 0], ancestors)][:, 1].unique())]
                
                # Usually only one parent
                for parent in parents:
                    curr_negative_idx = (~mask & masks[parent]).nonzero().squeeze(-1)
                    if len(curr_negative_idx) == 0:
                        continue
                    
                    # If already get some negative samples from its child, use them as positive samples for itself
                    if last_negative_sample is None:
                        positive_idx = mask.nonzero().squeeze(-1)
                        positive_samples.append(torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples)]
                                                            for _ in range(2)], dim=-1))
                    else:
                        positive_idx = last_negative_idx
                        positive_samples.append(last_negative_sample[:self.num_seg_samples])
                    
                    curr_negative_sample = torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)],
                                                    curr_negative_idx[np.random.choice(curr_negative_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)]], dim=-1)
                    negative_samples.append(curr_negative_sample)
                    _get_hierarchical_samples(masks[parent], masks, parent, curr_negative_sample[:self.num_seg_samples], curr_negative_idx)
                    
            else:
                assert last_negative_sample is not None and last_negative_idx is not None
                positive_idx = last_negative_idx
                negative_idx = (~mask).nonzero().squeeze(-1)
                positive_samples.append(last_negative_sample[:self.num_seg_samples])
                negative_samples.append(torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)],
                                                    negative_idx[np.random.choice(negative_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)]], dim=-1))
            
        if self.hierarchical_sampling:
            positive_samples = []
            negative_samples = []
            
            # Shape = (N_masks, 2) [child_idx, parent_idx]
            seg_hierarchy = seg_hierarchy.nonzero()
                        
            for i, mask in enumerate(masks):
                positive_idx = mask.nonzero().squeeze(-1)
                negative_idx = (~mask).nonzero().squeeze(-1)
                positive_samples.append(torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples)] 
                                                     for _ in range(2)], dim=-1))
                negative_samples.append(torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)],
                                                        negative_idx[np.random.choice(negative_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)]], dim=-1))
                
                # If has child
                if i in seg_hierarchy[:, 1]:
                    children = seg_hierarchy[(seg_hierarchy[:, 1] == i)][:, 0].unique()
                    
                    # Sample positive from parent & ~child
                    positive_idx = (mask & (~torch.any(masks[children], dim=0))).nonzero().squeeze(-1)
                    if positive_idx.numel() < FILTER_THRES * 2:
                        continue
                    
                    # Sample negative from ~parent
                    negative_idx = (~mask).nonzero().squeeze(-1)
                    positive_samples.append(torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples)] 
                                                         for _ in range(2)], dim=-1))
                    negative_samples.append(torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)],
                                                        negative_idx[np.random.choice(negative_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)]], dim=-1))
                    
                # If is the leaf node (no child, but has parent)
                elif i in seg_hierarchy[:, 0]:
                    _get_hierarchical_samples(mask, masks, i, None, None)      
            
            positive_samples = torch.stack(positive_samples)
            negative_samples = torch.stack(negative_samples)
        else:
            positive_idxs = [mask.nonzero().squeeze(-1) for mask in masks]
            negative_idxs = [(~mask).nonzero().squeeze(-1) for mask in masks]
            
            positive_samples = torch.stack([torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples)] 
                                            for positive_idx in positive_idxs]) for _ in range(2)], dim=-1)
            negative_samples = torch.stack([torch.stack([positive_idx[np.random.choice(positive_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)] 
                                            for positive_idx in positive_idxs]), 
                                           torch.stack([negative_idx[np.random.choice(negative_idx.shape[0], self.num_seg_samples * self.neg_sample_ratio)] 
                                            for negative_idx in negative_idxs])], dim=-1)
        
        return positive_samples, negative_samples
    
    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            if self.patch_size is None:
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            else:
                pix_idxs = self.sample_patch(self.img_wh[1], self.img_wh[0])

            rays = self.rays[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'rgb': rays[:, :3], 'img_wh': self.img_wh}

            if hasattr(self, 'seg') and len(self.seg):
                if self.ray_sampling_strategy == 'all_images':
                    # TODO
                    raise NotImplementedError
                elif self.ray_sampling_strategy == 'same_image':
                    with torch.no_grad():
                        if len(self.seg[img_idxs]) > 0:
                            segmentation_masks = self.seg[img_idxs][:, pix_idxs] # list of N, 4096
                            
                            total_pix_num = len(pix_idxs)
                            filter_mask = torch.logical_and(FILTER_THRES < segmentation_masks.count_nonzero(dim=1),
                                                             segmentation_masks.count_nonzero(dim=1) < total_pix_num - FILTER_THRES)
                            
                            segmentation_masks = segmentation_masks[filter_mask]
                            seg_hierarchy = self.seg_hierarchy[img_idxs][0][filter_mask][:, filter_mask]
                            
                            positive_samples, negative_samples = self.get_seg_samples(segmentation_masks, seg_hierarchy)

                            sample['seg_positives'] = positive_samples
                            sample['seg_negatives'] = negative_samples
                            sample['seg_masks'] = segmentation_masks
                            
                            if self.load_depth_smooth:
                                # 
                                leaf_seg_masks = self.seg[img_idxs][filter_mask][seg_hierarchy.sum(0) == 0]
                                # seg_maps = seg_maps.reshape(-1, self.img_wh[1], self.img_wh[0])
                                depth_smooth_pix_idxs = []
                                
                                for mask in leaf_seg_masks:
                                    true_indices = mask.nonzero().squeeze(1)
                                    sampled_true_indices = true_indices[np.random.choice(true_indices.shape[0], 16)]
                                    keep = ((sampled_true_indices % self.img_wh[0] != 0) & (sampled_true_indices % self.img_wh[0] < self.img_wh[0] - 2) & 
                                            (sampled_true_indices // self.img_wh[0] != 0) & (sampled_true_indices // self.img_wh[0] < self.img_wh[1] - 2))
                                    sampled_true_indices = sampled_true_indices[keep]
                                    
                                    vertical_keep = mask[sampled_true_indices - self.img_wh[0]] & mask[sampled_true_indices + self.img_wh[0]] & mask[sampled_true_indices + 2 * self.img_wh[0]] 
                                    horizontal_keep = mask[sampled_true_indices - 1] & mask[sampled_true_indices + 1] & mask[sampled_true_indices + 2]
                                    
                                    vertical_pix_idxs = torch.stack((sampled_true_indices[vertical_keep] - self.img_wh[0], 
                                                                sampled_true_indices[vertical_keep],
                                                                sampled_true_indices[vertical_keep] + self.img_wh[0],
                                                                sampled_true_indices[vertical_keep] + 2 * self.img_wh[0]), dim=1)
                                    horizontal_pix_idxs = torch.stack((sampled_true_indices[horizontal_keep] - 1, 
                                                                sampled_true_indices[horizontal_keep],
                                                                sampled_true_indices[horizontal_keep] + 1,
                                                                sampled_true_indices[horizontal_keep] + 2), dim=1)
                                        
                                    depth_smooth_pix_idxs += [vertical_pix_idxs.cpu().numpy(), horizontal_pix_idxs.cpu().numpy()]
                                    
                                depth_smooth_pix_idxs = np.concatenate(depth_smooth_pix_idxs).reshape(-1)
                                
                                # not_in_background = (~(self.rays[img_idxs, depth_smooth_pix_idxs][..., :3] > 0.99).all(dim=-1)).all(dim=-1)
                                # depth_smooth_pix_idxs = depth_smooth_pix_idxs[not_in_background].reshape(-1)
                                pix_idxs = np.concatenate((pix_idxs, depth_smooth_pix_idxs))
                                sample['pix_idxs'] = pix_idxs
                                sample['depth_smooth_samples_num'] = len(pix_idxs) - self.batch_size
                                
                        else:
                            sample['seg_positives'] = None
                            sample['seg_negatives'] = None
                            sample['seg_masks'] = None
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
            with torch.no_grad():
                if len(self.seg) > idx and len(self.seg[idx]) > 0:
                    pix_idxs = np.arange(0, self.img_wh[0]*self.img_wh[1])
                    segmentation_masks = self.seg[idx][:, pix_idxs] # list of N, 4096
                    
                    total_pix_num = len(pix_idxs)
                    filter_small = 5
                    filter_mask = torch.logical_and(filter_small < segmentation_masks.count_nonzero(dim=1),
                                                        segmentation_masks.count_nonzero(dim=1) < total_pix_num - filter_small)
                    
                    segmentation_masks = segmentation_masks[filter_mask]
                    
                    filter_mask = torch.logical_and(FILTER_THRES < segmentation_masks.count_nonzero(dim=1),
                                                        segmentation_masks.count_nonzero(dim=1) < total_pix_num - FILTER_THRES)
                    
                    segmentation_masks = segmentation_masks[filter_mask]
                    positive_idxs = [segmentation_masks.nonzero().squeeze(-1) for segmentation_masks in segmentation_masks]
                    negative_idxs = [(~segmentation_masks).nonzero().squeeze(-1) for segmentation_masks in segmentation_masks]
                    
                    positive_samples = torch.stack([torch.stack([positive[np.random.choice(positive.shape[0], self.num_seg_samples)] 
                                                    for positive in positive_idxs]) for _ in range(2)], dim=-1)

                    negative_samples = torch.stack([torch.stack([positive[np.random.choice(positive.shape[0], self.num_seg_samples)] 
                                                            for positive in positive_idxs]), torch.stack([negative[np.random.choice(negative.shape[0], self.num_seg_samples)] 
                                                            for negative in negative_idxs])], dim=-1)
                                                
                    sample['seg_positives'] = positive_samples
                    sample['seg_negatives'] = negative_samples
                    sample['seg_masks'] = None
                    
        sample['intrinsic'] = self.K
        
        return sample
