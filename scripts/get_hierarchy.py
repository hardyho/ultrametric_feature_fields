import imageio
import numpy as np
import os
import cv2
import tqdm
import argparse

IN_THRESHOLD = 0.95
SAME_THRESHOLD = 0.9

parser = argparse.ArgumentParser(
    description=(
        "Get hierarchy information"
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the segmentation folder",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the output hierarchy folder"
    ),
)

def get_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection/(mask1_area+mask2_area-intersection)
    in_mask1 = intersection/mask1_area
    in_mask2 = intersection/mask2_area
    return iou, in_mask1, in_mask2

    
def main(args):
    
    src_dir = args.input
    tgt_dir = args.output
    os.makedirs(tgt_dir, exist_ok=True)

    for folder in tqdm.tqdm(sorted(os.listdir(src_dir))):
        print(folder)
        mask_names = [m for m in sorted(os.listdir(os.path.join(src_dir, folder))) if m.endswith('png')]
        masks = [np.array(imageio.imread(os.path.join(src_dir, folder, mask)), dtype=bool) for mask in mask_names]
        iou = np.zeros((len(masks), len(masks)))
        in_mask = np.zeros((len(masks), len(masks)))
        keep = [True for _ in range(len(masks))]

        for i in range(len(masks)):
            for j in range(i, len(masks)):
                iou_, in_mask1, in_mask2 = get_iou(masks[i], masks[j])
                in_mask[i, j] = in_mask1
                in_mask[j, i] = in_mask2
                iou[i, j] = iou_
                iou[j, i] = iou_

        repeat = iou > SAME_THRESHOLD
        inside = np.logical_and(in_mask > IN_THRESHOLD,  iou < SAME_THRESHOLD)
        
        np.save(os.path.join(tgt_dir, folder + '_same.npy'), repeat)
        np.save(os.path.join(tgt_dir, folder + '_inside.npy'), inside)   
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)