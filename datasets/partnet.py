import torch
import json
import numpy as np
import os
from tqdm import tqdm
import glob
from .ray_utils import get_ray_directions

from .color_utils import read_image, read_seg_map
from .base import BaseDataset

def get_camera_position(alpha, beta, scale):
    x = np.sin(alpha * np.pi / 180) * scale
    z = np.cos(alpha * np.pi / 180) * np.cos(beta * np.pi / 180) * scale
    y = np.cos(alpha * np.pi / 180) * np.sin(beta * np.pi / 180) * scale
    return np.array([x, y, z])


def look_at(eye, target, up):
    """
    Compute the world to camera transform!
    """
    z_axis = (target - eye) / np.linalg.norm(target - eye)
    x_axis = np.cross(up, z_axis) / np.linalg.norm(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    z_axis = -z_axis  # for right hand rule!
    R = np.column_stack((x_axis, y_axis, z_axis))  # each vector becomes a column
    # todo: this a left-handed coordinate system?!

    # merge into view matrix
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = R
    view_matrix[:3, 3] = eye
    return view_matrix

class PartNetDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, len_per_epoch=1000, **kwargs):
        super().__init__(root_dir, split, downsample, len_per_epoch)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        IMG_SIZE = 224
        camera_angle = 50 * np.pi / 180
        w, h= int(IMG_SIZE * self.downsample), int(IMG_SIZE * self.downsample)
        f = w / (2 * np.tan(0.5 * camera_angle))
        K = np.float32([[f * self.downsample, 0, w/2],
                        [0, f * self.downsample, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)
        

    def read_meta(self, split, **kwargs):
        self.num_seg_samples = kwargs.get('num_seg_samples', 64) 
        self.neg_sample_ratio = kwargs.get('neg_sample_ratio', 1)
        self.render_train = kwargs.get('render_train', False)
        self.render_train_subsample = kwargs.get('render_train_subsample', 2)
        self.rotate_test = kwargs.get('rotate_test', False)
        self.load_depth_smooth = kwargs.get('load_depth_smooth', False)
        self.hierarchical_sampling = kwargs.get('hierarchical_sampling', False)
        self.num_training_frames = 0
                
        viewpoint_folders = sorted(os.listdir(str(self.root_dir)))

        img_paths = []
        poses = []
        
        for folder in viewpoint_folders:
            if not folder.startswith("view"):
                continue

            # load image
            fname = os.path.join(self.root_dir, folder, "shape-rgb.png")
            img_paths.append(fname)

            # load pose
            pose = np.loadtxt(os.path.join(self.root_dir, folder, "meta.txt"))
            z_rot, y_rot, x_rot = pose[:3]
            radius = pose[3]
            cam_center = get_camera_position(y_rot, z_rot, radius)

            # get W2C matrix with pytorch3D
            pose = look_at(cam_center, np.zeros(3), up=np.array([1, 0, 0]))
            pose[:, 1:3] *= -1
            poses.append(pose)

        self.poses = torch.FloatTensor(poses)[:, :3]  # camera to world transform
        scale_factor = 2.0
        self.poses[:, :3, 3] *= scale_factor
        self.scale_factor = scale_factor
        self.rays = []
        
        if kwargs.get('load_seg', False):
            # TODO: on-the-fly encoding
            seg_folder = os.path.join(self.root_dir, 'seg')
            seg_hierarchy_folder = seg_folder.replace('seg', 'seg_hierarchy')
            seg_paths = [os.path.join(seg_folder, name.split('/')[-2][-2:]) for name in img_paths]
            print("seg_paths", seg_paths)
        else:
            seg_paths = []
            
        
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            feature_paths = None
            seg_paths = [x for i, x in enumerate(seg_paths) if i%8!=0]
            self.poses = [x for i, x in enumerate(self.poses) if i%8!=0]
        elif split=='val':
            if self.render_train:
                img_paths = ([x for i, x in enumerate(img_paths) if i%8!=0][::self.render_train_subsample] 
                             + [x for i, x in enumerate(img_paths) if i%8 == 0])
                feature_paths = None
                seg_paths = ([x for i, x in enumerate(seg_paths) if i%8!=0][::self.render_train_subsample] 
                             + [x for i, x in enumerate(seg_paths) if i%8 == 0])
                self.poses = ([x for i, x in enumerate(self.poses) if i%8!=0][::self.render_train_subsample]
                             + [x for i, x in enumerate(self.poses) if i%8 == 0])
                self.num_training_frames = len([_ for i, x in enumerate(img_paths) if i%8!=0][::self.render_train_subsample])
            else:
                img_paths = [x for i, x in enumerate(img_paths) if i%8 == 0]
                feature_paths = None
                seg_paths = [x for i, x in enumerate(seg_paths) if i%8 == 0]
                self.poses = [x for i, x in enumerate(self.poses) if i%8 == 0]
            
        print(f'Loading {len(img_paths)} {split} images ...')
        self.features = []
        self.seg = []
        self.seg_hierarchy = []
        for i, (img_path) in enumerate(tqdm(img_paths)):
            buf = [] # buffer for ray attributes: rgb, etc
            print(img_path)
            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            buf += [img]

            self.rays += [torch.cat(buf, 1)]

            if feature_paths:
                self.features.append(torch.load(feature_paths[i]))
            
            if seg_paths:
                self.seg.append(torch.tensor([read_seg_map(os.path.join(seg_paths[i], name), self.img_wh) for name in sorted(os.listdir(seg_paths[i])) 
                                if name.endswith('png')], dtype=bool))                
                self.seg_hierarchy.append((torch.tensor(np.load(os.path.join(seg_hierarchy_folder, img_path.split('/')[-2][-2:] + '_inside.npy'))),
                                        torch.tensor(np.load(os.path.join(seg_hierarchy_folder, img_path.split('/')[-2][-2:] + '_same.npy')))))

        self.rays = torch.stack(self.rays)
        self.poses = torch.stack(self.poses) # (N_images, 3, 4)