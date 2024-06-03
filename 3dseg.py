import torch
import numpy as np
import scipy.sparse as sp 
import struct
import open3d.ml.torch as ml3d
import imageio
import os
import argparse
import tqdm
import glob
import open3d as o3d

parser = argparse.ArgumentParser()

parser.add_argument('exp', type=str)           # positional argument
parser.add_argument('output', type=str)           # positional argument

parser.add_argument('-t', '--threshold', type=float, default=5e-3)      # option that takes a value
parser.add_argument('--k_query', type=int, default=5)  # on/off flag
parser.add_argument('-o', '--n_outliers', type=int, default=2)  # on/off flag
args = parser.parse_args()


def get_segmentation(feat, k_index, distance, num_keep):
    num_points, k = k_index.shape
    rows = torch.arange(num_points).repeat(k)
    cols = k_index.transpose(0, 1).reshape(-1)

    data = (((feat[rows] - feat[cols]) ** 2).sum(-1) + 1e-8).sqrt()
    rows = rows[data < distance]
    cols = cols[data < distance]
    data = data[data < distance]

    graph = sp.csr_matrix((data.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())), shape=(len(feat), len(feat)))
    num_components, segmentation = sp.csgraph.connected_components(graph, connection='weak')
    
    count = np.histogram(segmentation, bins=[i for i in range(segmentation.max() + 2)])[0]
    segmentation = count.argsort()[::-1].argsort()[segmentation]
    segmentation[segmentation > num_keep] = num_keep # keep the num_keep largest masks
    
    return segmentation

def write_pointcloud(filename,xyz_points,rgb_points=None):
    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    print((rgb_points.sum(1) > 0).sum())
    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%(rgb_points.sum(1) > 0).sum(), 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        if  rgb_points[i].sum() > 0:
            fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                            rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                            rgb_points[i,2].tostring())))
    fid.close()

print('Start 3D segmentation')
coords = []
feats = []
query_coords = []
query_depths = []

last_epoch = max([int(f) for f in os.listdir(f'results/nerf/{args.exp}')])

num_training_frames = len(glob.glob(f'results/nerf/{args.exp}/{last_epoch}/train_*_d.npy'))
num_test_frames = len(glob.glob(f'results/nerf/{args.exp}/{last_epoch}/[0-9]*_d.npy'))

for i in tqdm.tqdm([i for i in range(num_training_frames)]):
    depth = np.load(f'results/nerf/{args.exp}/{last_epoch}/train_{i:03d}_d.npy').reshape(-1)
    coords.append(np.load(f'results/nerf/{args.exp}/{last_epoch}/train_{i:03d}_surface.npy').reshape(-1, 3)[depth > 0.3])
    feats.append(torch.load(f'results/nerf/{args.exp}/{last_epoch}/train_{i:03d}_f.pth')[depth > 0.3])
    
print('Loading test data')
for i in [i for i in range(num_test_frames)]:
    query_coords.append(np.load(f'results/nerf/{args.exp}/{last_epoch}/{i:03d}_surface.npy').reshape(-1, 3))
    query_depths.append(torch.tensor(np.load(f'results/nerf/{args.exp}/{last_epoch}/{i:03d}_d.npy')).reshape(-1))

coords = np.concatenate(coords)
feats = torch.cat(feats)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)

ind = np.array([i for i in range(len(coords))])

downsample_ind = [i[0] for i in pcd.voxel_down_sample_and_trace(2e-3, coords.min(axis=0), coords.max(axis=0))[2]]
print(f'Downsampling, {len(downsample_ind)}/{len(ind)} points left')
pcd_new = o3d.geometry.PointCloud()
pcd_new.points = o3d.utility.Vector3dVector(coords[downsample_ind])
pcd_new, ind = pcd_new.remove_radius_outlier(nb_points=1, radius=4e-3)

print(f'Cleaning, {len(ind)}/{len(downsample_ind)} points left')

U, S, V = torch.pca_lowrank(feats[downsample_ind][ind].float(), niter=10)
proj_V = V[:, :3].float()
lowrank = torch.matmul(feats[downsample_ind][ind].float(), proj_V)
lowrank = ((lowrank - lowrank.min(0, keepdim=True)[0]) / 
           (lowrank.max(0, keepdim=True)[0] - lowrank.min(0, keepdim=True)[0])).clip(0, 1)

os.makedirs('vis/', exist_ok=True)
write_pointcloud(f'vis/{args.output}.ply', coords[downsample_ind][ind], (lowrank.numpy() * 255).astype(np.uint8))

feats = feats[downsample_ind][ind]
coords = coords[downsample_ind][ind]
points = torch.tensor(coords)
k_graph = 17
num_seg = 200
k_query = args.k_query

nsearch = ml3d.layers.KNNSearch(return_distances=False)
nsearch_w_distance = ml3d.layers.KNNSearch(return_distances=True)
ans = nsearch(points, points, k_graph)
k_index = ans.neighbors_index.reshape(-1, k_graph)[:, 1:].long()

os.makedirs(f'vis/{args.output}/', exist_ok=True)
import tqdm

for distance in tqdm.tqdm([(i + 1) * 0.01 for i in range(0, 50)]):
    seg_result = get_segmentation(feats, k_index, distance, num_seg)
    coords_save = coords[ (seg_result!=num_seg)]
    seg_result_save = seg_result[ (seg_result!=num_seg)]
    for i in range(num_seg):
        lowrank[seg_result==i] = lowrank[seg_result==i].mean(dim=0)
    
    colors = np.random.randint(0, 256, (num_seg + 2, 3))
    colors[-2] = 0
    colors[-1] = 255
    for i in range(num_seg):
        colors[i] = (lowrank[seg_result==i] * 255).mean(dim=0)
    
    for i, (query_coord, query_depth) in enumerate(zip(query_coords, query_depths)):
        os.makedirs(f'vis/{args.output}/{i}', exist_ok=True)
        nn_ans = nsearch_w_distance(points[(seg_result!=num_seg)], torch.tensor(query_coord), k_query)
        nn_index = nn_ans.neighbors_index.reshape(-1, k_query).long()
        nn_distance = nn_ans.neighbors_distance.reshape(-1,k_query).max(dim=1)[0]
        segmentation_2d = seg_result_save[nn_index]
        
        segmentation_2d = torch.tensor(segmentation_2d).mode(dim=1)[0]
        segmentation_2d[nn_distance > args.threshold] = num_seg
        
        segmentation_2d = segmentation_2d.reshape(200, 200).numpy()
        np.save(f'vis/{args.output}/{i}/{distance:.03f}.npy', segmentation_2d)
        segmentation_2d = colors[segmentation_2d].astype(np.uint8)
        imageio.imsave(f'vis/{args.output}/{i}/{distance:.03f}.png', segmentation_2d)