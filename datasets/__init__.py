from .nerf import NeRFDataset
from .partnet import PartNetDataset

dataset_dict = {'nerf': NeRFDataset,
                'partnet': PartNetDataset}   