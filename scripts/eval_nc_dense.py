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

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str)           # positional argument
parser.add_argument('name', type=str)           # positional argument

args = parser.parse_args()

def get_miou(gt_mask, masks):
    miou = (gt_mask & masks).sum(axis=(1,2)) / (gt_mask | masks).sum(axis=(1,2))
    # print(miou)
    return max(miou)

   
json_dict = {}

dataset_nc_scores_0 = []
dataset_nc_scores_1 = []
dataset_nc_scores_2 = []

with open(f'{args.root_dir}/transforms_val.json') as f:
    image_names = [frame['file_path'].split('/')[-1] for frame in json.load(f)['frames']]
    
for i, image_name in tqdm(enumerate(sorted(image_names, key=lambda x: int(x[2:])))):
    print(image_name)

    gt_seg_0 = np.array(imageio.imread(f'{args.root_dir}/gt_seg/val/{image_name}_seg_objects.tif'))[..., 0]
    gt_seg_1 = np.array(imageio.imread(f'{args.root_dir}/gt_seg/val/{image_name}_seg_collections.tif'))[..., 0]
    gt_seg_2 = gt_seg_1 > 0
    
    seg_maps_path = f'vis/{args.name}/{i}'
    seg_maps = np.array([np.array(imageio.imread(os.path.join(seg_maps_path, s))) for s in os.listdir(seg_maps_path) if s.endswith('png')])
    seg_maps = np.array([np.load(os.path.join(seg_maps_path, s)) for s in os.listdir(seg_maps_path) if s.endswith('npy')])

    label_num = np.max(seg_maps) + 1
    one_hot_seg_map = np.eye(label_num, dtype=bool)[seg_maps].transpose(0,3,1,2).reshape(-1, *gt_seg_1.shape)
    
    nc_scores_0 = []
    for seg_id in [i for i in np.unique(gt_seg_0) if i != 0]:
        if (gt_seg_0 == seg_id).sum() < 20:
            pass
        else:
            nc_scores_0.append(get_miou(gt_seg_0 == seg_id, one_hot_seg_map))
    
    print(nc_scores_0, sum(nc_scores_0)/len(nc_scores_0))
    dataset_nc_scores_0.append(sum(nc_scores_0)/len(nc_scores_0))
    
    nc_scores_1 = []
    for seg_id in [i for i in np.unique(gt_seg_1) if i != 0]:
        if (gt_seg_1 == seg_id).sum() < 20:
            pass
        else:
            nc_scores_1.append(get_miou(gt_seg_1 == seg_id, one_hot_seg_map))
    
    print(nc_scores_1, sum(nc_scores_1)/len(nc_scores_1))
    dataset_nc_scores_1.append(sum(nc_scores_1)/len(nc_scores_1))
    
    nc_scores_2 = []
    for seg_id in [i for i in np.unique(gt_seg_2)]:
        if (gt_seg_2 == seg_id).sum() < 20:
            pass
        else:
            nc_scores_2.append(get_miou(gt_seg_2 == seg_id, one_hot_seg_map))
    
    print(nc_scores_2, sum(nc_scores_2)/len(nc_scores_2))
    dataset_nc_scores_2.append(sum(nc_scores_2)/len(nc_scores_2))
    
    json_dict[image_name] = {
        'nc_scores_0': sum(nc_scores_0)/len(nc_scores_0),
        'nc_scores_1': sum(nc_scores_1)/len(nc_scores_1),
        'nc_scores_2': sum(nc_scores_2)/len(nc_scores_2),
    }
    
json_dict['nc_scores_0'] = sum(dataset_nc_scores_0)/len(dataset_nc_scores_0)
json_dict['nc_scores_1'] = sum(dataset_nc_scores_1)/len(dataset_nc_scores_1)
json_dict['nc_scores_2'] = sum(dataset_nc_scores_2)/len(dataset_nc_scores_2)

json_dict['nc_scores_mean'] = (json_dict['nc_scores_0'] + json_dict['nc_scores_1'] + json_dict['nc_scores_2']) / 3


os.makedirs(f'eval_results/{args.name}', exist_ok=True)
with open(f'eval_results/{args.name}/nc_score.json', 'w') as f:
    json.dump(json_dict, f)
    