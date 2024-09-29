# Custom dataset

## Usage
1. Follow the instruction in SAM(https://github.com/facebookresearch/segment-anything) to download the checkpoint and install SAM with 
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
We use the vit-h model to generate our data.

2. Run SAM to get the segmentation data:
```
python seg.py --input $DATASET_PATH/{train, val} --output $DATASET_PATH/{train, val}_seg --model-type vit_h --checkpoint $CKPT_PATH --resize-fact $RESIZE_FACT$
```
This will generate segmentation labels in the following structure:
```
{train, val}_seg
    - IMG_NAME
        - MASK_NAME
        ...
    ...
```
We recommend resizing the image to the size used for NeRF training before running SAM to make the data processing more efficient.

3. Get the segmentation hierarchy with:
```
python get_hierarchy.py --input $DATASET_PATH/{train, val}_seg --output $DATASET_PATH/{train, val}_seg_hierarchy
```
