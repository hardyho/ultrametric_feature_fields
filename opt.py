import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'partnet'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument("--accumulate_grad_batches", type=int, default=None, help="number of steps of gradient accumulation")

    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    parser.add_argument('--feature_directory', type=str, default=None)
    parser.add_argument('--feature_dim', type=int, default=None)
    
    parser.add_argument('--render_feature', action='store_true', default=False,
                        help='use volumn rendering to get the feature map')
    
    
    parser.add_argument('--ultrametric_weight', type=float, default=0.0,
                        help='loss weight for the Ultrametric loss')
    parser.add_argument('--euclidean_weight', type=float, default=0.0,
                        help='loss weight for the Euclidean loss')
    
    parser.add_argument('--num_seg_samples', type=int, default=64, 
                        help='number of data pairs sampling for each mask')
    parser.add_argument('--neg_sample_ratio', type=int, default=1,
                        help='ratio of negative samples to positive samples in segmentatin data sampling')
    parser.add_argument('--num_seg_test', type=int, default=200,
                        help='number of segmentation masks during 2D segentation inference')
    
    parser.add_argument('--load_seg', action='store_true', default=False,
                        help='load segmentation data')
    parser.add_argument('--depth_smooth', action='store_true', default=False,
                        help='use depth smoothing loss')
    parser.add_argument('--hierarchical_sampling', action='store_true', default=False,
                        help='use hierarchical sampling')
    parser.add_argument('--run_seg_inference', action='store_true', default=False,
                        help='run 2D segentation inference')
    parser.add_argument('--render_train', action='store_true', default=False,
                        help='render images and features in the training set, needed for 3D segmentation')
    parser.add_argument('--render_train_subsample', type=int, default=2,
                        help='subsample ratio when rendering training set, set to higher ratio to save Disk Storage and time')
    parser.add_argument('--rotate_test', action='store_true', default=False,
                        help='render images with additional rotation for view consistency evaluation')

    return parser.parse_args()
