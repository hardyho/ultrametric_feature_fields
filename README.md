# View-Consistent Hierarchical 3D Segmentation Using Ultrametric Feature Fields

[Paper](https://www.arxiv.org/pdf/2405.19678) | [Code](https://github.com/hardyho/ultrametric_feature_fields)

This is the codebase of "View-Consistent Hierarchical 3D Segmentation Using Ultrametric Feature Fields".


https://github.com/hardyho/ultrametric_feature_fields/assets/61956100/3ec7968b-db14-4527-b5c4-8fa1b2e0c8b0


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

### Blender-HS Dataset
Download the Blender-HS dataset from the [link](https://drive.google.com/file/d/1b7PPaZ8QTGF_lsv8rqvSKm3-IWMbK-c9/view?usp=sharing). Organized the dataset in following structure:
```
-
  - train
    - IMAGE_i.png
        ...
  - val
  - train_seg
    - IMAGE_i
      - MASK_i.png
      ...
  - val_seg
  - train_seg_hierarchy
    ...
  - val_seg_hierarchy
  - train_depth
  - val_depth
  - val_visible
  - gt_seg
    - train
    - val
  - transforms_train.json
  - transforms_val.json
  - transforms_val_rotate_90.json
```

For custom datasets, you can follow this [instruction](scripts/README.md) to generate SAM outputs and the get the hierarchy information (`{train,val}_seg` and `{train,val}_seg_hierarchy`). 

### PartNet Dataset
Download the PartNet Dataset with rendered images in three catagories, chair, table, cabinet, following [this](https://github.com/mikacuy/joint_learning_retrieval_deformation#data-download-and-preprocessing-details)(3.Targets > Images). Generate SAM outputs (see above) and the get the hierarchy information for the dataset. The layout should be:
```
-
  - seg
  - seg_hierarchy
  - view-00
    - meta.txt
    - part-XXX.png
    - shape-rgb.png
  - view-01
  ...
```

## Usage

### NeRF Training
```
python train.py --root_dir ROOT_DIR --dataset_name nerf --exp_name EXP_NAME --render_feature --downsample 0.25 --num_epochs 20 --batch_size 4096 --ray_sampling_strategy same_image --feature_dim 128 --load_seg --hierarchical_sampling --ultrametric_weight 1.0 --euclidean_weight 1.0 --num_seg_samples 64 --depth_smooth --lr 1e-2 --run_seg_inference --render_train --rotate_test
```

- `--root_dir` is the root directory of the dataset.
- `--ultrametric_weight` and `--euclidean_weight` is the loss weight of the contrastive loss in the Ultrametric and the Euclidean space respectively
- `--run_seg_inference` will generate per-frame 2D scene parsing result during the inference (which may be view-inconsistent). This is just for visualization and are not needed for generating 3D segmentation.
- `--render_train` will render the feature maps of images in the training set which are used in 3D segmentation. 
- `--rotate_test` will render rotated test views for view consistency evaluation

### PartNet Training
```
python train.py --root_dir ROOT_DIR --dataset_name nerf --exp_name EXP_NAME --render_feature --downsample 0.25 --num_epochs 20 --batch_size 4096 --ray_sampling_strategy same_image --feature_dim 128 --load_seg --hierarchical_sampling --ultrametric_weight 1.0 --euclidean_weight 1.0 --num_seg_samples 64 --depth_smooth --lr 1e-2 --run_seg_inference --render_train
```

### (Optional) NeRF Inference
```
python render.py --root_dir ROOT_DIR --dataset_name nerf --exp_name EXP_NAME --render_feature --downsample 0.25 --num_epochs 20 --batch_size 4096 --ray_sampling_strategy same_image --feature_dim 128 --load_seg --hierarchical_sampling --ultrametric_weight 1.0 --euclidean_weight 1.0 --num_seg_samples 64 --depth_smooth --lr 1e-2 --run_seg_inference --render_train --rotate_test --ckpt_path CKPT_PATH --render_dir RENDER_DIR
```

Run `render.py` if you want to run inference with checkpoints again after training. We recommend you to set `RENDER_DIR` as `results/nerf/NAME` for 3D segmentation.

### 3D Segmentation
```
python 3dseg.py EXP_NAME OUTPUT_NAME
```

Run the command to get the 3D segmentation results, which includes a 3D point clouds colored by segmentation id, and view-consistent 2D segmentation maps for all test views.

### Evaluation
```
python scripts/eval_nc.py ROOT_DIR EXP_NAME
python scripts/eval_vc.py ROOT_DIR EXP_NAME
```

Run evalution to get the Normal Covering Score and the View Consistency Score of the 3D segmentation result.

## Citation

The codebase of NeRF is derived from [ngp_pl](https://github.com/kwea123/ngp_pl/commit/6b2a66928d032967551ab98d5cd84c7ef1b83c3d) and [Distilled Feature Fields](https://github.com/pfnet-research/distilled-feature-fields)

If you find this project useful in your research, please cite:
```
@misc{he2024viewconsistent,
      title={View-Consistent Hierarchical 3D SegmentationUsing Ultrametric Feature Fields}, 
      author={Haodi He and Colton Stearns and Adam W. Harley and Leonidas J. Guibas},
      year={2024},
      eprint={2405.19678},
      archivePrefix={arXiv},
}
```
