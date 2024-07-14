import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image, read_seg_map

from .base import BaseDataset


class NeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, len_per_epoch=1000, **kwargs):
        super().__init__(root_dir, split, downsample, len_per_epoch)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
            meta = json.load(f)

        w = h = int(800*self.downsample)
        fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])*self.downsample

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split, **kwargs):
        self.rays = []
        self.poses = []
        
        self.num_seg_samples = kwargs.get('num_seg_samples', 64) 
        self.neg_sample_ratio = kwargs.get('neg_sample_ratio', 1)
        self.render_train = kwargs.get('render_train', False)
        self.render_train_subsample = kwargs.get('render_train_subsample', 2)
        self.rotate_test = kwargs.get('rotate_test', False)
        self.load_depth_smooth = kwargs.get('load_depth_smooth', False)
        self.hierarchical_sampling = kwargs.get('hierarchical_sampling', False)
        self.num_training_frames = 0
            
        if split == 'trainval':
            with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
                frames = json.load(f)["frames"]
            with open(os.path.join(self.root_dir, "transforms_val.json"), 'r') as f:
                frames += json.load(f)["frames"]
        elif self.rotate_test and split == 'val':
            frames = (json.load(open(os.path.join(
                self.root_dir, "transforms_val.json"), 'r'))['frames'] +
                json.load(open(os.path.join(
                    self.root_dir, "transforms_val_rotate_90.json"), 'r'))['frames'])
        else:
            with open(os.path.join(self.root_dir, f"transforms_{split}.json"), 'r') as f:
                frames = json.load(f)["frames"]
        
        if self.render_train and split == 'val':
            with open(os.path.join(self.root_dir, f"transforms_train.json"), 'r') as f:
                training_frames = json.load(f)["frames"][::self.render_train_subsample]
                self.num_training_frames = len(training_frames)
                frames = training_frames + frames

        if kwargs.get('load_seg', False):
            # TODO: on-the-fly encoding
            seg_folder =  os.path.join(self.root_dir, f'{split}_seg')
            seg_hierarchy_folder = os.path.join(self.root_dir, f'{split}_seg_hierarchy')
            seg_paths = [os.path.join(seg_folder, os.path.splitext(frame['file_path'])[0].split('/')[-1]) for frame in frames]
        else:
            seg_paths = []
            
        self.features = []
        self.seg = []
        self.seg_hierarchy = []
            
        print(f'Loading {len(frames)} {split} images ...')
        for i, frame in tqdm(enumerate(frames)):
            c2w = np.array(frame['transform_matrix'])[:3, :4]

            # determine scale
            if 'Jrender_Dataset' in self.root_dir:
                c2w[:, :2] *= -1 # [left up front] to [right down front]
                folder = self.root_dir.split('/')
                scene = folder[-1] if folder[-1] != '' else folder[-2]
                if scene=='Easyship':
                    pose_radius_scale = 1.2
                elif scene=='Scar':
                    pose_radius_scale = 1.8
                elif scene=='Coffee':
                    pose_radius_scale = 2.5
                elif scene=='Car':
                    pose_radius_scale = 0.8
                else:
                    pose_radius_scale = 1.5
            else:
                c2w[:, 1:3] *= -1 # [right up back] to [right down front]
                pose_radius_scale = 1.5
            c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale

            # add shift
            if 'Jrender_Dataset' in self.root_dir:
                if scene=='Coffee':
                    c2w[1, 3] -= 0.4465
                elif scene=='Car':
                    c2w[0, 3] -= 0.7
            self.poses += [c2w]

            
            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = read_image(img_path, self.img_wh)
            self.rays += [img]
            
            if seg_paths:
                self.seg.append(torch.tensor([read_seg_map(os.path.join(seg_paths[i], name), self.img_wh) for name in sorted(os.listdir(seg_paths[i])) 
                                    if name.endswith('png')], dtype=bool))
                self.seg_hierarchy.append((torch.tensor(np.load(os.path.join(seg_hierarchy_folder, img_path.split('/')[-1][:-4] + '_inside.npy'))),
                                            torch.tensor(np.load(os.path.join(seg_hierarchy_folder, img_path.split('/')[-1][:-4] + '_same.npy')))))
                        

        if len(self.rays)>0:
            self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
            
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
