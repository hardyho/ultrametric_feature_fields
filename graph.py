import torch
import numpy as np
from itertools import repeat
import higra as hg

def get_ultrametric(feat, rays, data_pairs, seg_masks=None, visualize=False, pix_ids=None, img_wh=None):
    # data pairs : positive pairs idx 
    N, C = feat.shape
    Num_masks, Num_pairs, D = data_pairs.shape
    assert D == 2
    
    pixs_distances = ((rays.unsqueeze(0) - rays.unsqueeze(1)) ** 2).sum(-1)
    pixs_topk = torch.topk(pixs_distances, 11, largest=False)[1][:, 1:]
    
    data_pairs = data_pairs.reshape(-1, 2)
    srcs = np.array([i for i in range(len(pixs_topk)) for _ in repeat(None, 10)])
    tgts = pixs_topk.reshape(-1)
    
    data = (feat[srcs] * feat[tgts]).sum(-1)
    graph_edge_lengths =  1 - data
    
    if seg_masks is not None:
        on_boundary = torch.any(seg_masks[:, srcs] != seg_masks[:, tgts], dim=0)
        graph_edge_lengths[on_boundary] += 10
    
    tree, altitudes = hg.bpt_canonical((srcs, tgts.cpu().numpy(), len(feat)), graph_edge_lengths.detach().cpu().numpy().astype(float))
    tree.lowest_common_ancestor_preprocess()
    edge_idx = np.zeros(data_pairs.shape[0], dtype=np.int64)
    for i, (v1, v2) in enumerate(data_pairs):
        edge_idx[i] = tree.lowest_common_ancestor(v1, v2)
    
    edge_idx[edge_idx < N] = N
    edge_idx = edge_idx - N
    mst_map = tree.mst_edge_map
    edge_idx = mst_map[edge_idx]
        
    ultrametric = data[edge_idx]
    ultrametric[data_pairs[:, 0] == data_pairs[:, 1]] = 1
    ultrametric = ultrametric.reshape(Num_masks, Num_pairs)

    return ultrametric