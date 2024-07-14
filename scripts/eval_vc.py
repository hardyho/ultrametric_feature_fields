import imageio
import numpy as np
import os, json
from tqdm import tqdm
import torch
import argparse
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_DIR)

from datasets.color_utils import read_image, read_seg_map
from datasets.ray_utils import get_ray_directions, get_rays

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str)           # positional argument
parser.add_argument('name', type=str)           # positional argument

args = parser.parse_args()

def get_miou(gt_mask, masks):
    miou = (gt_mask & masks).sum(axis=(1,2)) / (gt_mask | masks).sum(axis=(1,2))
    return max(miou), np.argmax(miou)

def get_trasnform(transform_matrix_1, transform_matrix_2, gt_depth):
    pose_radius_scale = 1.5
    c2w_1 = np.array(transform_matrix_1)[:3, :4]
    c2w_1[:, 1:3] *= -1
    c2w_1[:, 3] /= np.linalg.norm(c2w_1[:, 3])/pose_radius_scale
    c2w_1 = torch.FloatTensor(c2w_1)
    
    c2w_2 = np.array(transform_matrix_2)[:3, :4]
    c2w_2[:, 1:3] *= -1
    c2w_2[:, 3] /= np.linalg.norm(c2w_2[:, 3])/pose_radius_scale
    c2w_2 = torch.FloatTensor(c2w_2)
     
    rays_o_1, rays_d = get_rays(directions, c2w_1)
    rays_o_2, _ = get_rays(directions, c2w_2)
   
    gt_surface = (gt_depth.unsqueeze(-1) * rays_d + rays_o_1)
    frame_2_camera_space = (gt_surface - rays_o_2) @ c2w_2[..., :3]
    frame_2_screen_space = frame_2_camera_space[..., :2] / frame_2_camera_space[..., 2].unsqueeze(-1)
    
    transform = (frame_2_screen_space * fx - 0.5 + 100).int().clip(0, w-1)[..., (1, 0)]

    return transform
   
json_dict = {}

dataset_vc_scores_0 = []
dataset_vc_scores_1 = []
dataset_vc_scores_2 = []

with open(f'{args.root_dir}/transforms_val_rotate_90.json', 'r') as f:
    data = json.load(f)
    rotated_frames = data['frames']
    
fx = fy = 0.5 * 200 / np.tan(0.5*data['camera_angle_x'])
w = h = 200
K = np.float32([[fx, 0, w/2],
                [0, fy, h/2],
                [0,  0,   1]])
K = torch.FloatTensor(K)
directions = get_ray_directions(h, w, K)

with open(f'{args.root_dir}/transforms_val.json') as f:
    frames = json.load(f)['frames']
    image_names = [frame['file_path'].split('/')[-1] for frame in frames]
    
for i, image_name in tqdm(enumerate(sorted(image_names, key=lambda x: int(x[2:])))):
    print(image_name)

    gt_seg_0 = np.array(imageio.imread(f'{args.root_dir}/gt_seg/val/{image_name}_seg_objects.tif'))[..., 0]
    gt_seg_1 = np.array(imageio.imread(f'{args.root_dir}/gt_seg/val/{image_name}_seg_collections.tif'))[..., 0]
    gt_seg_2 = gt_seg_1 > 0
    
    seg_maps_path = f'vis/{args.name}/{i}'
    seg_maps = np.array([np.load(os.path.join(seg_maps_path, s)) for s in sorted(os.listdir(seg_maps_path)) if s.endswith('npy')])
    
    seg_maps_rotated_path = f'vis/{args.name}/{i + len(frames)}'
    seg_maps_rotated = np.array([np.load(os.path.join(seg_maps_rotated_path, s)) for s in sorted(os.listdir(seg_maps_rotated_path)) if s.endswith('npy')])

    visible_src_mask = np.array(imageio.imread(f'{args.root_dir}/val_visible_90/{image_name}_rotate.png'))[-200:, -200:, 0].astype(bool)
    visible_tgt_mask = np.array(imageio.imread(f'{args.root_dir}/val_visible_90/{image_name}.png'))[-200:, -200:, 0].astype(bool)

    label_num = np.max(seg_maps) + 1

    frame = frames[i]
    rotated_frame = rotated_frames[i]
    depth_img = imageio.imread(f'{args.root_dir}/val_depth/{image_name}_depth.png')[..., 0].reshape(-1) / 256
    gt_depth = 8 * (1 - torch.FloatTensor(depth_img)) / 2.687419 

    transform = get_trasnform(frame['transform_matrix'],
                              rotated_frame['transform_matrix'],
                              gt_depth
                            )


    vc_scores_0 = []
    for seg_id in [i for i in np.unique(gt_seg_0) if i != 0]:
        if (gt_seg_0 == seg_id).sum() < 20:
            pass
        else:
            _vc_scores = []
            for _ in range(100):
                sampled_point = np.array((gt_seg_0 == seg_id).nonzero())[:, np.random.randint(0, (gt_seg_0 == seg_id).sum())]
                segmentation_masks = (seg_maps[:, sampled_point[0]:sampled_point[0]+1, 
                                    sampled_point[1]:sampled_point[1]+1] == seg_maps)
                miou, best_matched_id = get_miou(gt_seg_0 == seg_id, segmentation_masks)

                masks = seg_maps[best_matched_id]
                masks_rotated = seg_maps_rotated[best_matched_id]

                # First, find the corresponding mask in the rotated view
                sampleing_mask = ((gt_seg_0 == seg_id) & visible_tgt_mask & 
                                   visible_src_mask[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)] &
                                   (masks_rotated[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)] != 200))
                if ~sampleing_mask.any():
                    continue

                sampled_point = np.array(sampleing_mask.nonzero())[:, np.random.randint(0, sampleing_mask.sum())]
                sampled_point_rotated = transform.reshape(h,w,2)[sampled_point[0], sampled_point[1]]
                mask_rotated = (masks_rotated == masks_rotated[sampled_point_rotated[0], sampled_point_rotated[1]])

                # Then, use the ground truth depth to project it back to the original view
                mask_rotated_rotated = mask_rotated[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)]
                mask_rotated_rotated[~visible_tgt_mask] = False
                        
                # Compare it with the mask in the original view
                mask = (masks == masks[sampled_point[0], sampled_point[1]])
                mask[~visible_tgt_mask] = False

                miou = (mask & mask_rotated_rotated).sum() / (mask | mask_rotated_rotated).sum()
                _vc_scores.append(miou)  

            if len(_vc_scores):
                vc_scores_0.append(sum(_vc_scores)/len(_vc_scores))
    
    print(vc_scores_0, sum(vc_scores_0)/len(vc_scores_0))
    dataset_vc_scores_0.append(sum(vc_scores_0)/len(vc_scores_0))
    
    vc_scores_1 = []
    for seg_id in [i for i in np.unique(gt_seg_1) if i != 0]:
        if (gt_seg_1 == seg_id).sum() < 20:
            pass
        else:
            _vc_scores = []
            for _ in range(100):
                sampled_point = np.array((gt_seg_1 == seg_id).nonzero())[:, np.random.randint(0, (gt_seg_1 == seg_id).sum())]
                segmentation_masks = (seg_maps[:, sampled_point[0]:sampled_point[0]+1, 
                                    sampled_point[1]:sampled_point[1]+1] == seg_maps)
                miou, best_matched_id = get_miou(gt_seg_1 == seg_id, segmentation_masks)

                masks = seg_maps[best_matched_id]
                masks_rotated = seg_maps_rotated[best_matched_id]

                # First, find the corresponding mask in the rotated view
                sampleing_mask = ((gt_seg_1 == seg_id) & visible_tgt_mask & 
                                   visible_src_mask[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)] &
                                   (masks_rotated[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)] != 200))
                if ~sampleing_mask.any():
                    continue

                sampled_point = np.array(sampleing_mask.nonzero())[:, np.random.randint(0, sampleing_mask.sum())]
                sampled_point_rotated = transform.reshape(h,w,2)[sampled_point[0], sampled_point[1]]
                mask_rotated = (masks_rotated == masks_rotated[sampled_point_rotated[0], sampled_point_rotated[1]])

                # Then, use the ground truth depth to project it back to the original view
                mask_rotated_rotated = mask_rotated[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)]
                mask_rotated_rotated[~visible_tgt_mask] = False
                        
                # Compare it with the mask in the original view
                mask = (masks == masks[sampled_point[0], sampled_point[1]])
                mask[~visible_tgt_mask] = False

                miou = (mask & mask_rotated_rotated).sum() / (mask | mask_rotated_rotated).sum()
                _vc_scores.append(miou)  

            if len(_vc_scores):
                vc_scores_1.append(sum(_vc_scores)/len(_vc_scores))
    
    print(vc_scores_1, sum(vc_scores_1)/len(vc_scores_1))
    dataset_vc_scores_1.append(sum(vc_scores_1)/len(vc_scores_1))
    
    vc_scores_2 = []
    for seg_id in [i for i in np.unique(gt_seg_2)]:
        if (gt_seg_2 == seg_id).sum() < 20:
            pass
        else:
            _vc_scores = []
            for _ in range(100):
                sampled_point = np.array((gt_seg_2 == seg_id).nonzero())[:, np.random.randint(0, (gt_seg_2 == seg_id).sum())]
                segmentation_masks = (seg_maps[:, sampled_point[0]:sampled_point[0]+1, 
                                    sampled_point[1]:sampled_point[1]+1] == seg_maps)
                miou, best_matched_id = get_miou(gt_seg_2 == seg_id, segmentation_masks)

                masks = seg_maps[best_matched_id]
                masks_rotated = seg_maps_rotated[best_matched_id]

                # First, find the corresponding mask in the rotated view
                sampleing_mask = ((gt_seg_2 == seg_id) & visible_tgt_mask & 
                                   visible_src_mask[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)] &
                                   (masks_rotated[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)] != 200))
                if ~sampleing_mask.any():
                    continue

                sampled_point = np.array(sampleing_mask.nonzero())[:, np.random.randint(0, sampleing_mask.sum())]
                sampled_point_rotated = transform.reshape(h,w,2)[sampled_point[0], sampled_point[1]]
                mask_rotated = (masks_rotated == masks_rotated[sampled_point_rotated[0], sampled_point_rotated[1]])

                # Then, use the ground truth depth to project it back to the original view
                mask_rotated_rotated = mask_rotated[transform[...,0].reshape(h,w), transform[...,1].reshape(h,w)]
                mask_rotated_rotated[~visible_tgt_mask] = False
                        
                # Compare it with the mask in the original view
                mask = (masks == masks[sampled_point[0], sampled_point[1]])
                mask[~visible_tgt_mask] = False

                miou = (mask & mask_rotated_rotated).sum() / (mask | mask_rotated_rotated).sum()
                _vc_scores.append(miou)  

            if len(_vc_scores):
                vc_scores_2.append(sum(_vc_scores)/len(_vc_scores))

    print(vc_scores_2, sum(vc_scores_2)/len(vc_scores_2))
    dataset_vc_scores_2.append(sum(vc_scores_2)/len(vc_scores_2))
    
    json_dict[image_name] = {
        'vc_scores_0': sum(vc_scores_0)/len(vc_scores_0),
        'vc_scores_1': sum(vc_scores_1)/len(vc_scores_1),
        'vc_scores_2': sum(vc_scores_2)/len(vc_scores_2),
    }
    
json_dict['vc_scores_0'] = sum(dataset_vc_scores_0)/len(dataset_vc_scores_0)
json_dict['vc_scores_1'] = sum(dataset_vc_scores_1)/len(dataset_vc_scores_1)
json_dict['vc_scores_2'] = sum(dataset_vc_scores_2)/len(dataset_vc_scores_2)

json_dict['vc_scores_mean'] = (json_dict['vc_scores_0'] + json_dict['vc_scores_1'] + json_dict['vc_scores_2']) / 3


os.makedirs(f'eval_results/{args.name}', exist_ok=True)
with open(f'eval_results/{args.name}/vc_score.json', 'w') as f:
    json.dump(json_dict, f)
    