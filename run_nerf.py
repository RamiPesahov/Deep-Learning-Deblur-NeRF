import os
import time

import cv2
import imageio
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn

from NeRF import *
from load_llff import load_llff_data
from run_nerf_helpers import *
from metrics import compute_img_metric

from radam import RAdam
from loss import sigma_sparsity_loss, total_variation_loss

DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', required=True, help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, required=True, help='input data directory')
    parser.add_argument("--datadownsample", type=float, default=-1,
                        help='if downsample > 0, means downsample the image to scale=datadownsample')
    parser.add_argument("--tbdir", type=str, required=True, help="tensorboard log directory")
    parser.add_argument("--num_gpu", type=int, default=1, help=">1 will use DataParallel")
    parser.add_argument("--torch_hub_dir", type=str, default='', help="torch hub cache dir")
    parser.add_argument("--num_input_views", type=int, default=-1,
                    help="number of LLFF views to use; -1 means use all available views")

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 64,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 128,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=50000, help='number of iteration')
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')

    # hash / ngp-like options
    parser.add_argument("--i_embed", type=int, default=0,
                        help='2=spherical, 1=hash encoding, 0=positional encoding, -1=none')
    parser.add_argument("--i_embed_views", type=int, default=0,
                        help='2=spherical, 1=hash encoding, 0=positional encoding, -1=none')
    parser.add_argument("--finest_res", type=int, default=512, help='finest resolution for hashed embedding')
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help='log2 of hashmap size')
    parser.add_argument("--sparse_loss_weight", type=float, default=0)
    parser.add_argument("--tv_loss_weight", type=float, default=0.0)
    parser.add_argument("--multi_optimizer", type=int, default=0,
                        help='0=one optimizer, 1=separate ngp/deblur optimizers')

    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for viewdir encoding')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='sigma noise std')
    parser.add_argument("--rgb_activate", type=str, default='sigmoid', help='rgb activation')
    parser.add_argument("--sigma_activate", type=str, default='relu', help='sigma activation')

    # kernel options
    parser.add_argument("--kernel_type", type=str, default='deformablesparsekernel',
                        help='choose among <none>, <itsampling>, <sparsekernel>')
    parser.add_argument("--kernel_isglobal", action='store_true', help='if specified, canonical kernel is global')
    parser.add_argument("--kernel_start_iter", type=int, default=0, help='start training kernel after # iteration')
    parser.add_argument("--kernel_ptnum", type=int, default=5, help='number of sparse locations in the kernel')
    parser.add_argument("--kernel_random_hwindow", type=float, default=0.25,
                        help='randomly displace the predicted ray position')
    parser.add_argument("--kernel_img_embed", type=int, default=32, help='dim of image latent code')
    parser.add_argument("--kernel_rand_dim", type=int, default=2, help='dimensions of input random number')
    parser.add_argument("--kernel_rand_embed", type=int, default=3, help='embed frequency of kernel coordinate')
    parser.add_argument("--kernel_rand_mode", type=str, default='float', help='<float>, <int#>, <fix>')
    parser.add_argument("--kernel_random_mode", type=str, default='input', help='<input>, <output>')
    parser.add_argument("--kernel_spatial_embed", type=int, default=0, help='dim of spatial coordinate embedding')
    parser.add_argument("--kernel_depth_embed", type=int, default=0, help='dim of depth coordinate embedding')
    parser.add_argument("--kernel_hwindow", type=int, default=10, help='max window of the kernel')
    parser.add_argument("--kernel_pattern_init_radius", type=float, default=0.1, help='init radius of pattern')
    parser.add_argument("--kernel_num_hidden", type=int, default=3, help='number of hidden layers')
    parser.add_argument("--kernel_num_wide", type=int, default=64, help='hidden width')
    parser.add_argument("--kernel_shortcut", action='store_true', help='if yes, add a shortcut to the network')
    parser.add_argument("--kernel_topk", type=int, default=3,
                        help='render only the top-K kernel samples by learned weight; 0 disables pruning')

    # occupancy-grid acceleration (Instant-NGP style, conservative)
    parser.add_argument("--occ_grid_res", type=int, default=96,
                        help='resolution of occupancy grid; 0 disables it')
    parser.add_argument("--occ_update_every", type=int, default=16,
                        help='update occupancy grid every N training steps')
    parser.add_argument("--occ_warmup_steps", type=int, default=1024,
                        help='do not prune before this many steps')
    parser.add_argument("--occ_decay", type=float, default=0.95,
                        help='EMA decay for occupancy grid')
    parser.add_argument("--occ_threshold", type=float, default=0.01,
                        help='occupancy threshold for keeping a sample')
    parser.add_argument("--occ_density_threshold", type=float, default=0.01,
                        help='density threshold to mark a voxel occupied')
    parser.add_argument("--occ_anchor_interval", type=int, default=8,
                        help='always keep every Nth sample along ray as anchor')

    parser.add_argument("--align_start_iter", type=int, default=0, help='start iteration of the align loss')
    parser.add_argument("--align_end_iter", type=int, default=int(1e10), help='end iteration of the align loss')
    parser.add_argument("--kernel_align_weight", type=float, default=0, help='align term weight')

    parser.add_argument("--prior_start_iter", type=int, default=0, help='start iteration of the prior loss')
    parser.add_argument("--prior_end_iter", type=int, default=int(1e10), help='end iteration of the prior loss')
    parser.add_argument("--kernel_prior_weight", type=float, default=0, help='prior loss weight')

    parser.add_argument("--sparsity_start_iter", type=int, default=0, help='start iteration of sparsity loss')
    parser.add_argument("--sparsity_end_iter", type=int, default=int(1e10), help='end iteration of sparsity loss')
    parser.add_argument("--kernel_sparsity_type", type=str, default='tv', choices=['tv', 'normalize', 'robust'])
    parser.add_argument("--kernel_sparsity_weight", type=float, default=0, help='weight of sparsity loss')

    parser.add_argument("--kernel_spatialvariant_trans", action='store_true',
                        help='optimize spatial variant 3D translation of each sampling point')
    parser.add_argument("--kernel_global_trans", action='store_true',
                        help='optimize global 3D translation of each sampling point')
    parser.add_argument("--tone_mapping_type", type=str, default='none',
                        help='tone mapping of linear to LDR color space')

    # render options
    parser.add_argument("--render_only", action='store_true', help='reload weights and render only')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render path')
    parser.add_argument("--render_multipoints", action='store_true',
                        help='render sub image that reconstructs the blur image')
    parser.add_argument("--render_rmnearplane", type=int, default=0,
                        help='set density of nearest plane to 0 when render')
    parser.add_argument("--render_focuspoint_scale", type=float, default=1., help='scale the focal point')
    parser.add_argument("--render_radius_scale", type=float, default=1., help='scale the camera radius')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor for fast preview')
    parser.add_argument("--render_epi", action='store_true', help='render the video with epi path')

    # llff flags
    parser.add_argument("--factor", type=int, default=None, help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates')
    parser.add_argument("--lindisp", action='store_true', help='sample linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set')

    # compatibility options
    parser.add_argument("--precrop_iters", type=int, default=0)
    parser.add_argument("--precrop_frac", type=float, default=.5)
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8)
    parser.add_argument("--shape", type=str, default='greek')
    parser.add_argument("--white_bkgd", action='store_true')
    parser.add_argument("--half_res", action='store_true')

    # logging/saving
    parser.add_argument("--i_print", type=int, default=200)
    parser.add_argument("--i_tensorboard", type=int, default=200)
    parser.add_argument("--i_weights", type=int, default=20000)
    parser.add_argument("--i_testset", type=int, default=20000)
    parser.add_argument("--i_video", type=int, default=20000)

    return parser


def build_optimizers(args, nerf):
    if args.i_embed == 1 and args.multi_optimizer == 0:
        params = list(nerf.mlp_coarse.parameters())
        if args.N_importance > 0 and nerf.mlp_fine is not None:
            params += list(nerf.mlp_fine.parameters())
        if args.kernel_type == 'deformablesparsekernel' and nerf.kernelsnet is not None:
            params += list(nerf.kernelsnet.parameters())
        if args.tone_mapping_type == 'learn':
            params += list(nerf.tonemapping.parameters())

        embedding_params = list(nerf.embed_fn.parameters())
        optimizer = RAdam(
            [
                {'params': params, 'weight_decay': 1e-6},
                {'params': embedding_params, 'eps': 1e-15},
            ],
            lr=args.lrate,
            betas=(0.9, 0.99)
        )
        return optimizer, None, None

    elif args.i_embed == 1 and args.multi_optimizer == 1:
        nerf_params = list(nerf.mlp_coarse.parameters())
        if args.N_importance > 0 and nerf.mlp_fine is not None:
            nerf_params += list(nerf.mlp_fine.parameters())
        embedding_params = list(nerf.embed_fn.parameters())

        deblur_params = []
        if args.kernel_type == 'deformablesparsekernel' and nerf.kernelsnet is not None:
            deblur_params += list(nerf.kernelsnet.parameters())
        if args.tone_mapping_type == 'learn':
            deblur_params += list(nerf.tonemapping.parameters())

        ngp_optimizer = RAdam(
            [
                {'params': nerf_params, 'weight_decay': 1e-6},
                {'params': embedding_params, 'eps': 1e-15},
            ],
            lr=0.01,
            betas=(0.9, 0.99)
        )

        deblur_optimizer = torch.optim.Adam(
            params=deblur_params,
            lr=5e-4,
            betas=(0.9, 0.999)
        ) 
        return None, ngp_optimizer, deblur_optimizer

    else:
        optimizer = torch.optim.Adam(
            params=nerf.parameters(),
            lr=args.lrate,
            betas=(0.9, 0.999)
        )
        return optimizer, None, None


def save_checkpoint(path, global_step, nerf, args, optimizer=None, ngp_optimizer=None, deblur_optimizer=None):
    save_dict = {
        'global_step': global_step,
        'network_state_dict': nerf.state_dict(),
    }

    if args.multi_optimizer == 0:
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
    else:
        if ngp_optimizer is not None:
            save_dict['ngp_optimizer_state_dict'] = ngp_optimizer.state_dict()
        if deblur_optimizer is not None:
            save_dict['deblur_optimizer_state_dict'] = deblur_optimizer.state_dict()

    torch.save(save_dict, path)


def maybe_load_checkpoint(args, basedir, expname, nerf, optimizer=None, ngp_optimizer=None, deblur_optimizer=None):
    start = 0

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpt_dir = os.path.join(basedir, expname)
        ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir)) if f.endswith('.tar')]

    print('Found ckpts', ckpts)

    if len(ckpts) == 0 or args.no_reload:
        return start

    ckpt_path = ckpts[-1]
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    start = ckpt['global_step']

    if args.multi_optimizer == 0:
        if optimizer is not None and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    else:
        if ngp_optimizer is not None and 'ngp_optimizer_state_dict' in ckpt:
            ngp_optimizer.load_state_dict(ckpt['ngp_optimizer_state_dict'])
        if deblur_optimizer is not None and 'deblur_optimizer_state_dict' in ckpt:
            deblur_optimizer.load_state_dict(ckpt['deblur_optimizer_state_dict'])

    smart_load_state_dict(nerf, ckpt)
    return start


def update_learning_rates(args, global_step, optimizer=None, ngp_optimizer=None, deblur_optimizer=None):
    decay_rate = 0.1

    if args.multi_optimizer == 0:
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
    else:
        ngp_decay_steps = 10 * 1000
        new_ngp_lrate = 0.01 * (decay_rate ** (global_step / ngp_decay_steps))
        if ngp_optimizer is not None:
            for param_group in ngp_optimizer.param_groups:
                param_group['lr'] = new_ngp_lrate

        deblur_decay_steps = 250 * 1000 
        new_deblur_lrate = 5e-4 * (decay_rate ** (global_step / deblur_decay_steps)) 
        if deblur_optimizer is not None:
            for param_group in deblur_optimizer.param_groups:
                param_group['lr'] = new_deblur_lrate


def train():
    parser = config_parser()
    args = parser.parse_args()

    if len(args.torch_hub_dir) > 0:
        print(f"Change torch hub cache to {args.torch_hub_dir}")
        torch.hub.set_dir(args.torch_hub_dir)

    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(
            args, args.datadir, args.factor,
            recenter=True, bd_factor=.75,
            spherify=args.spherify,
            path_epi=args.render_epi
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        args.bounding_box = bounding_box
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        print('LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.min(bds) * 0.9
            far = np.max(bds) * 1.0
        else:
            near = 0.0
            far = 1.0
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    imagesf = images
    images = (images * 255).astype(np.uint8)
    images_idx = np.arange(0, len(images))

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses)

    basedir = args.basedir
    tensorboardbase = args.tbdir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(tensorboardbase, expname), exist_ok=True)

    tensorboard = SummaryWriter(os.path.join(tensorboardbase, expname))

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write(f'{arg} = {attr}\n')

    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

        with open(test_metric_file, 'a') as file:
            file.write(open(args.config, 'r').read())
            file.write("\n============================\n||\n\\/\n")

    if args.kernel_type == 'deformablesparsekernel':
        kernelnet = DSKnet(
            args, len(images), torch.tensor(poses[:, :3, :4]),
            args.kernel_ptnum, args.kernel_hwindow,
            random_hwindow=args.kernel_random_hwindow,
            in_embed=args.kernel_rand_embed,
            random_mode=args.kernel_random_mode,
            img_embed=args.kernel_img_embed,
            spatial_embed=args.kernel_spatial_embed,
            depth_embed=args.kernel_depth_embed,
            num_hidden=args.kernel_num_hidden,
            num_wide=args.kernel_num_wide,
            short_cut=args.kernel_shortcut,
            pattern_init_radius=args.kernel_pattern_init_radius,
            isglobal=args.kernel_isglobal,
            optim_trans=args.kernel_global_trans,
            optim_spatialvariant_trans=args.kernel_spatialvariant_trans
        )
    elif args.kernel_type == 'none':
        kernelnet = None
    else:
        raise RuntimeError(f"kernel_type {args.kernel_type} not recognized")

    nerf = NeRFAll(args, kernelnet)
    print("embed_fn type:", type(nerf.embed_fn))
    print("mlp_coarse type:", type(nerf.mlp_coarse))
    if nerf.mlp_fine is not None:
        print("mlp_fine type:", type(nerf.mlp_fine))

    if args.num_gpu > 1:
        nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))

    optimizer, ngp_optimizer, deblur_optimizer = build_optimizers(args, nerf)

    start = maybe_load_checkpoint(
        args, basedir, expname, nerf,
        optimizer=optimizer,
        ngp_optimizer=ngp_optimizer,
        deblur_optimizer=deblur_optimizer,
    )

    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.0

    bds_dict = {'near': near, 'far': far}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    global_step = start

    render_poses = torch.tensor(render_poses[:, :3, :4]).to(device)
    nerf = nerf.to(device)

    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            testsavedir = os.path.join(
                basedir, expname,
                f"renderonly_{'test' if args.render_test else 'path'}_{start:06d}"
            )
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")

            nerf.eval()

            if args.render_test and kernelnet is not None:
                all_poses = torch.cat([render_poses, dummy_poses], dim=0)
                all_indices = torch.arange(all_poses.shape[0], device=device)
                rgbshdr, disps = nerf(
                    hwf[0], hwf[1], K, args.chunk,
                    poses=all_poses,
                    images_indices=all_indices,
                    render_kwargs=render_kwargs_test,
                    render_factor=args.render_factor,
                )
            else:
                rgbshdr, disps = nerf(
                    hwf[0], hwf[1], K, args.chunk,
                    poses=torch.cat([render_poses, dummy_poses], dim=0),
                    render_kwargs=render_kwargs_test,
                    render_factor=args.render_factor,
                )

            rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
            disps = (1.0 - disps)
            disps = disps[:len(disps) - dummy_num].cpu().numpy()
            rgbs = to8b(rgbshdr.cpu().numpy())
            disps = to8b(disps / disps.max())

            if args.render_test:
                for rgb_idx, rgb8 in enumerate(rgbs):
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                    imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'), disps[rgb_idx])
            else:
                prefix = 'epi_' if args.render_epi else ''
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video.mp4'), rgbs, fps=30, quality=9)
                imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_disp.mp4'), disps, fps=30, quality=9)

            if args.render_test and args.render_multipoints:
                for pti in range(args.kernel_ptnum):
                    nerf.eval()
                    poses_num = len(poses) + dummy_num
                    imgidx = torch.arange(poses_num, dtype=torch.long, device=render_poses.device).reshape(poses_num, 1)
                    rgbs, weights = nerf(
                        hwf[0], hwf[1], K, args.chunk,
                        poses=torch.cat([render_poses, dummy_poses], dim=0),
                        render_kwargs=render_kwargs_test,
                        render_factor=args.render_factor,
                        render_point=pti,
                        images_indices=imgidx
                    )
                    rgbs = rgbs[:len(rgbs) - dummy_num]
                    weights = weights[:len(weights) - dummy_num]
                    rgbs = to8b(rgbs.cpu().numpy())
                    weights = to8b(weights.cpu().numpy())

                    for rgb_idx, rgb8 in enumerate(rgbs):
                        imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_pt{pti}.png'), rgb8)
                        imageio.imwrite(os.path.join(testsavedir, f'w_{rgb_idx:03d}_pt{pti}.png'), weights[rgb_idx])
        return

    N_rand = args.N_rand
    train_datas = {}

    if args.datadownsample > 0:
        images_train = np.stack(
            [cv2.resize(img_, None, None, 1 / args.datadownsample, 1 / args.datadownsample, cv2.INTER_AREA)
             for img_ in imagesf],
            axis=0
        )
    else:
        images_train = imagesf

    num_img, hei, wid, _ = images_train.shape
    print(f"train on image sequence of len = {num_img}, {wid}x{hei}")
    k_train = np.array([
        K[0, 0] * wid / W, 0, K[0, 2] * wid / W,
        0, K[1, 1] * hei / H, K[1, 2] * hei / H,
        0, 0, 1
    ]).reshape(3, 3).astype(K.dtype)

    print('get rays')
    rays = np.stack([get_rays_np(hei, wid, k_train, p) for p in poses[:, :3, :4]], 0)
    rays = np.transpose(rays, [0, 2, 3, 1, 4])
    train_datas['rays'] = rays[i_train].reshape(-1, 2, 3)

    xs, ys = np.meshgrid(np.arange(wid, dtype=np.float32), np.arange(hei, dtype=np.float32), indexing='xy')
    xs = np.tile((xs[None, ...] + HALF_PIX) * W / wid, [num_img, 1, 1])
    ys = np.tile((ys[None, ...] + HALF_PIX) * H / hei, [num_img, 1, 1])
    train_datas['rays_x'] = xs[i_train].reshape(-1, 1)
    train_datas['rays_y'] = ys[i_train].reshape(-1, 1)

    train_datas['rgbsf'] = images_train[i_train].reshape(-1, 3)

    images_idx_tile = images_idx.reshape((num_img, 1, 1))
    images_idx_tile = np.tile(images_idx_tile, [1, hei, wid])
    train_datas['images_idx'] = images_idx_tile[i_train].reshape(-1, 1).astype(np.int64)

    print('shuffle rays')
    shuffle_idx = np.random.permutation(len(train_datas['rays']))
    train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
    print('done')
    i_batch = 0

    images = torch.tensor(images).to(device)
    imagesf = torch.tensor(imagesf).to(device)
    poses = torch.tensor(poses).to(device)
    train_datas = {k: torch.tensor(v).to(device) for k, v in train_datas.items()}

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    for i in range(start, N_iters):
        time0 = time.time()

        iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in train_datas.items()}
        batch_rays = iter_data.pop('rays').permute(0, 2, 1)

        i_batch += N_rand
        if i_batch >= len(train_datas['rays']):
            print("Shuffle data after an epoch!")
            shuffle_idx = np.random.permutation(len(train_datas['rays']))
            train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
            i_batch = 0

        nerf.train()
        if i == args.kernel_start_iter and device != torch.device("cpu"):
            torch.cuda.empty_cache()

        rgb, rgb0, extra_loss, ngp_loss = nerf(
            H, W, K, chunk=args.chunk,
            rays=batch_rays, rays_info=iter_data,
            retraw=True,
            force_naive=i < args.kernel_start_iter,
            **render_kwargs_train
        )

        target_rgb = iter_data['rgbsf'].squeeze(-2)
        img_loss = img2mse(rgb, target_rgb)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if args.N_importance > 0 and rgb0 is not None:
            img_loss0 = img2mse(rgb0, target_rgb)
            loss = loss + img_loss0

        extra_loss = {k: torch.mean(v) for k, v in extra_loss.items()}
        if len(extra_loss) > 0:
            for k, v in extra_loss.items():
                if f"kernel_{k}_weight" in vars(args):
                    if vars(args)[f"{k}_start_iter"] <= i <= vars(args)[f"{k}_end_iter"]:
                        loss = loss + v * vars(args)[f"kernel_{k}_weight"]

        sparsity_loss = args.sparse_loss_weight * (
            ngp_loss['sparsity_loss'].sum() +
            ngp_loss.get("sparsity_loss0", torch.tensor(0.0, device=rgb.device)).sum()
        )
        loss = loss + sparsity_loss

        if args.i_embed == 1:
            embed_fn = nerf.module.embed_fn if isinstance(nerf, nn.DataParallel) else nerf.embed_fn

            if hasattr(embed_fn, "embeddings") and hasattr(embed_fn, "n_levels"):
                n_levels = embed_fn.n_levels
                min_res = embed_fn.base_resolution
                max_res = embed_fn.finest_resolution
                log2_hashmap_size = embed_fn.log2_hashmap_size

                TV_loss = sum(
                    total_variation_loss(
                        embed_fn.embeddings[level],
                        min_res, max_res,
                        level, log2_hashmap_size,
                        n_levels=n_levels
                    )
                    for level in range(n_levels)
                )

                loss = loss + args.tv_loss_weight * TV_loss
                if i > 1000:
                    args.tv_loss_weight = 0.0

        if args.multi_optimizer == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            ngp_optimizer.zero_grad()
            deblur_optimizer.zero_grad()
            loss.backward()
            ngp_optimizer.step()
            deblur_optimizer.step()

        update_learning_rates(
            args, global_step,
            optimizer=optimizer,
            ngp_optimizer=ngp_optimizer,
            deblur_optimizer=deblur_optimizer
        )

        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, f'{i:06d}.tar')
            save_checkpoint(
                path, global_step, nerf, args,
                optimizer=optimizer,
                ngp_optimizer=ngp_optimizer,
                deblur_optimizer=deblur_optimizer
            )
            print('Saved checkpoints at', path)

        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                nerf.eval()
                rgbs, disps = nerf(H, W, K, args.chunk, poses=render_poses, render_kwargs=render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, f'{expname}_spiral_{i:06d}_')
            rgbs = (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min() + 1e-8)
            rgbs = rgbs.cpu().numpy()
            disps = disps.cpu().numpy()
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / disps.max()), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, f'testset_{i:06d}')
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses.shape)

            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")

            with torch.no_grad():
                nerf.eval()

                if kernelnet is not None:
                    all_poses = torch.cat([poses, dummy_poses], dim=0).to(device)
                    all_indices = torch.arange(all_poses.shape[0], device=device)
                    rgbs, _ = nerf(
                        H, W, K, args.chunk,
                        poses=all_poses,
                        images_indices=all_indices,
                        render_kwargs=render_kwargs_test
                    )
                else:
                    rgbs, _ = nerf(
                        H, W, K, args.chunk,
                        poses=torch.cat([poses, dummy_poses], dim=0).to(device),
                        render_kwargs=render_kwargs_test
                    )

                rgbs = rgbs[:len(rgbs) - dummy_num]
                rgbs_save = rgbs

                for rgb_idx, rgb in enumerate(rgbs_save):
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, f'{rgb_idx:03d}.png')
                    imageio.imwrite(filename, rgb8)

                rgbs_eval = rgbs[i_test]
                target_rgb_ldr = imagesf[i_test]

                test_mse = compute_img_metric(rgbs_eval, target_rgb_ldr, 'mse')
                test_psnr = compute_img_metric(rgbs_eval, target_rgb_ldr, 'psnr')
                test_ssim = compute_img_metric(rgbs_eval, target_rgb_ldr, 'ssim')

                tensorboard.add_scalar("Test MSE", test_mse, global_step)
                tensorboard.add_scalar("Test PSNR", test_psnr, global_step)
                tensorboard.add_scalar("Test SSIM", test_ssim, global_step)

            with open(test_metric_file, 'a') as outfile:
                outfile.write(
                    f"iter{i}/globalstep{global_step}: "
                    f"MSE:{test_mse:.8f} PSNR:{test_psnr:.8f} SSIM:{test_ssim:.8f}\n"
                )
            print('Saved test set')

        if i % args.i_tensorboard == 0:
            tensorboard.add_scalar("Loss", loss.item(), global_step)
            tensorboard.add_scalar("PSNR", psnr.item(), global_step)
            for k, v in extra_loss.items():
                tensorboard.add_scalar(k, v.item(), global_step)

        if i % args.i_print == 0:
            dt = time.time() - time0
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Time: {dt:.4f}s")

        global_step += 1


if __name__ == '__main__':
    if device != torch.device("cpu"):
        torch.set_default_device('cuda')
    train()