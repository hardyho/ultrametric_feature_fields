# View-Consistent Hierarchical 3D Segmentation Using Ultrametric Feature Fields

This is the codebase of View-Consistent Hierarchical 3D Segmentation Using Ultrametric Feature Fields.

## Enviornment

Setup
```
# assume cuda 11.1
pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 --extra-index-url https://download.pytorch.org/whl/cu111 --no-cache-dir
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cu111.html

pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
git submodule update --init --recursive
cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..
pip install models/csrc/
```

## Data

Download the Blender-HS dataset from the [link](TODO). Organized the dataset in following structure:
```
-
  - train
  - val
  - train_seg
    - IMAGE_i.png
    ...
  - val_seg
  - train_seg_hierarchy
    - IMAGE_i
      - MASK_i.png
      ...
    ...
  - val_seg_hierarchy
  - transforms_train.json
  - transforms_val.json
```

The scripts for generating similar SAM outputs on new custom datasets is coming soon.

## Usage

### NeRF Training
```
python train.py --root_dir ROOT_DIR --dataset_name nerf --exp_name EXP_NAME --render_feature --downsample 0.25 --num_epochs 1 --batch_size 4096 --ray_sampling_strategy same_image --feature_dim 128 --load_seg --hierarchical_sampling --ultrametric_weight 1.0 --euclidean_weight 1.0 --num_seg_samples 64 --depth_smooth --lr 1e-2 --run_seg_inference --render_train
```

- `--root_dir` is the root directory of the dataset.
- `--ultrametric_weight` and `--euclidean_weight` is the loss weight of the contrastive loss in the Ultrametric and the Euclidean space respectively
- `--run_seg_inference` will generate per-frame 2D scene parsing result during the inference (which may be view-inconsistent). This is just for visualization and are not needed for generating 3D segmentation.
- `--render_train` will render the feature maps of images in the training set which are used in 3D segmentation. 

### 3D Segmentation
```
python 3dseg.py EXP_NAME OUTPUT_NAME
```

Run the command to get the 3D segmentation results, which includes a 3D point clouds colored by segmentation id, and view-consistent 2D segmentation maps for all test views.


## Citation

The codebase of NeRF is derived from [ngp_pl](https://github.com/kwea123/ngp_pl/commit/6b2a66928d032967551ab98d5cd84c7ef1b83c3d) and [Distilled Feature Fields](https://github.com/pfnet-research/distilled-feature-fields)

TODO: ADD BIBTEX