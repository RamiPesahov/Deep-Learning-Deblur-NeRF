import os
import time
import pdb
import imageio
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from run_nerf_helpers import *

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    tcnn = None
    HAS_TCNN = False

print("NeRF HAS_TCNN =", HAS_TCNN)


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class DSKnet(nn.Module):
    def __init__(self, args, num_img, poses, num_pt, kernel_hwindow, *, random_hwindow=0.25,
                 in_embed=3, random_mode='input', img_embed=32, spatial_embed=0, depth_embed=0,
                 num_hidden=3, num_wide=64, short_cut=False, pattern_init_radius=0.1,
                 isglobal=False, optim_trans=False, optim_spatialvariant_trans=False):
        super().__init__()
        self.args = args
        self.num_pt = num_pt
        self.num_img = num_img
        self.short_cut = short_cut
        self.kernel_hwindow = kernel_hwindow
        self.random_hwindow = random_hwindow
        self.random_mode = random_mode
        self.isglobal = isglobal
        pattern_num = 1 if isglobal else num_img
        assert self.random_mode in ['input', 'output'], \
            f"DSKNet::random_mode {self.random_mode} unrecognized, should be input/output"

        self.register_buffer("poses", poses)
        self.register_parameter(
            "pattern_pos",
            nn.Parameter(torch.randn(pattern_num, num_pt, 2).float() * pattern_init_radius, True)
        )

        self.optim_trans = optim_trans
        self.optim_sv_trans = optim_spatialvariant_trans

        if optim_trans:
            self.register_parameter(
                "pattern_trans",
                nn.Parameter(torch.zeros(pattern_num, num_pt, 2).float(), True)
            )

        if in_embed > 0:
            self.in_embed_fn, self.in_embed_cnl = get_embedder(in_embed, args, input_dim=2)
        else:
            self.in_embed_fn, self.in_embed_cnl = None, 0

        self.img_embed_cnl = img_embed

        if spatial_embed > 0:
            self.spatial_embed_fn, self.spatial_embed_cnl = get_embedder(spatial_embed, args, input_dim=2)
        else:
            self.spatial_embed_fn, self.spatial_embed_cnl = None, 0

        if depth_embed > 0:
            self.require_depth = True
            self.depth_embed_fn, self.depth_embed_cnl = get_embedder(depth_embed, args, input_dim=1)
        else:
            self.require_depth = False
            self.depth_embed_fn, self.depth_embed_cnl = None, 0

        in_cnl = self.in_embed_cnl + self.img_embed_cnl + self.depth_embed_cnl + self.spatial_embed_cnl
        out_cnl = 1 + 2 + 2 if self.optim_sv_trans else 1 + 2

        hiddens = [nn.Linear(num_wide, num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]

        self.linears = nn.Sequential(
            nn.Linear(in_cnl, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1 = nn.Sequential(
            nn.Linear((num_wide + in_cnl) if short_cut else num_wide, num_wide), nn.ReLU(),
            nn.Linear(num_wide, out_cnl)
        )

        self.linears.apply(init_linear_weights)
        self.linears1.apply(init_linear_weights)

        if img_embed > 0:
            self.register_parameter(
                "img_embed",
                nn.Parameter(torch.zeros(num_img, img_embed).float(), True)
            )
        else:
            self.img_embed = None

    def forward(self, H, W, K, rays, rays_info):
        img_idx = rays_info['images_idx'].squeeze(-1)
        img_embed = self.img_embed[img_idx] if self.img_embed is not None else \
            torch.tensor([], device=img_idx.device).reshape(len(img_idx), self.img_embed_cnl)

        pt_pos = self.pattern_pos.expand(len(img_idx), -1, -1) if self.isglobal else self.pattern_pos[img_idx]
        pt_pos = torch.tanh(pt_pos) * self.kernel_hwindow

        if self.random_hwindow > 0 and self.random_mode == "input":
            random_pos = torch.randn_like(pt_pos) * self.random_hwindow
            pt_pos = pt_pos + random_pos

        input_pos = pt_pos
        if self.in_embed_fn is not None:
            pt_pos = pt_pos * (np.pi / self.kernel_hwindow)
            pt_pos = self.in_embed_fn(pt_pos)

        img_embed_expand = img_embed[:, None].expand(len(img_embed), self.num_pt, self.img_embed_cnl)
        x = torch.cat([pt_pos, img_embed_expand], dim=-1)

        rays_x, rays_y = rays_info['rays_x'], rays_info['rays_y']
        if self.spatial_embed_fn is not None:
            spatialx = rays_x / (W / 2 / np.pi) - np.pi
            spatialy = rays_y / (H / 2 / np.pi) - np.pi
            spatial = torch.cat([spatialx, spatialy], dim=-1)
            spatial = self.spatial_embed_fn(spatial)
            spatial = spatial[:, None].expand(len(img_idx), self.num_pt, self.spatial_embed_cnl)
            x = torch.cat([x, spatial], dim=-1)

        if self.depth_embed_fn is not None:
            depth = rays_info['ray_depth']
            depth = depth * np.pi
            depth = self.depth_embed_fn(depth)
            depth = depth[:, None].expand(len(img_idx), self.num_pt, self.depth_embed_cnl)
            x = torch.cat([x, depth], dim=-1)

        x1 = self.linears(x)
        x1 = torch.cat([x, x1], dim=-1) if self.short_cut else x1
        x1 = self.linears1(x1)

        delta_trans = None
        if self.optim_sv_trans:
            delta_trans, delta_pos, weight = torch.split(x1, [2, 2, 1], dim=-1)
        else:
            delta_pos, weight = torch.split(x1, [2, 1], dim=-1)

        if self.optim_trans:
            delta_trans = self.pattern_trans.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]

        if delta_trans is None:
            delta_trans = torch.zeros_like(delta_pos)

        delta_trans = delta_trans * 0.01
        new_rays_xy = delta_pos + input_pos

        temperature = 0.1
        weight = torch.softmax(weight[..., 0] / temperature, dim=-1)

        if self.args.kernel_topk > 0 and self.args.kernel_topk < weight.shape[-1]:
            topk = self.args.kernel_topk
            topk_weight, topk_idx = torch.topk(weight, topk, dim=-1)

            new_rays_xy = torch.gather(
                new_rays_xy,
                dim=1,
                index=topk_idx.unsqueeze(-1).expand(-1, -1, new_rays_xy.shape[-1])
            )

            delta_trans = torch.gather(
                delta_trans,
                dim=1,
                index=topk_idx.unsqueeze(-1).expand(-1, -1, delta_trans.shape[-1])
            )

            weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-8)

        if self.random_hwindow > 0 and self.random_mode == 'output':
            raise NotImplementedError(f"{self.random_mode} for self.random_mode is not implemented")

        poses = self.poses[img_idx]

        rays_x = (rays_x - K[0, 2] + new_rays_xy[..., 0]) / K[0][0]
        rays_y = -(rays_y - K[1, 2] + new_rays_xy[..., 1]) / K[1][1]
        dirs = torch.stack([
            rays_x - delta_trans[..., 0],
            rays_y - delta_trans[..., 1],
            -torch.ones_like(rays_x)
        ], -1)

        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3], -1)

        translation = torch.stack([
            delta_trans[..., 0],
            delta_trans[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)
        rays_o = torch.sum(translation[..., None, :] * poses[:, None], dim=-1)

        align = new_rays_xy[:, 0, :].abs().mean()
        align += (delta_trans[:, 0, :].abs().mean() * 10)

        return torch.stack([rays_o, rays_d], dim=-1), weight, align


class NeRFAll(nn.Module):
    def __init__(self, args, kernelsnet=None):
        super().__init__()
        self.args = args
        self.occ_step = 0
        self.occ_grid = None
        self.occ_enabled = False
        self.occ_update_every = getattr(args, "occ_update_every", 16)
        self.occ_warmup_steps = getattr(args, "occ_warmup_steps", 1024)
        self.occ_decay = getattr(args, "occ_decay", 0.95)
        self.occ_threshold = getattr(args, "occ_threshold", 0.01)
        self.occ_density_threshold = getattr(args, "occ_density_threshold", 0.01)
        self.occ_anchor_interval = getattr(args, "occ_anchor_interval", 8)
        self.occ_grid_res = getattr(args, "occ_grid_res", 96)
        self.embed_fn, self.input_ch = get_embedder(args.multires, args, args.i_embed)
        self.input_ch_views = 0
        self.kernelsnet = kernelsnet
        self.embeddirs_fn = None

        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(
                args.multires_views, args, args.i_embed_views
            )

        self.output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]

        if args.i_embed == 1:
            if HAS_TCNN:
                print("Using tiny-cuda-nn small MLP")
                self.mlp_coarse = TCNNNeRFSmall(
                    input_ch=self.input_ch,
                    input_ch_views=self.input_ch_views,
                    hidden_dim=64,
                    geo_feat_dim=15,
                    num_layers=2,
                    num_layers_color=3,
                )
            else:
                self.mlp_coarse = NeRFSmall(
                    num_layers=2,
                    hidden_dim=64,
                    geo_feat_dim=15,
                    num_layers_color=3,
                    hidden_dim_color=64,
                    input_ch=self.input_ch,
                    input_ch_views=self.input_ch_views
                )
        else:
            self.mlp_coarse = NeRF(
                D=args.netdepth, W=args.netwidth,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs
            )

        self.mlp_fine = None
        if args.N_importance > 0:
            if args.i_embed == 1:
                if HAS_TCNN:
                    self.mlp_fine = TCNNNeRFSmall(
                        input_ch=self.input_ch,
                        input_ch_views=self.input_ch_views,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers=2,
                        num_layers_color=3,
                    )
                else:
                    self.mlp_fine = NeRFSmall(
                        num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=self.input_ch,
                        input_ch_views=self.input_ch_views
                    )
            else:
                self.mlp_fine = NeRF(
                    D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                    input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs
                )

        activate = {
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'none': lambda x: x,
            'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
            'softplus': lambda x: nn.Softplus()(x - 1)
        }
        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = ToneMapping(args.tone_mapping_type)

    def mlpforward(self, inputs, viewdirs, mlp, netchunk=1024 * 64):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        if HAS_TCNN and isinstance(mlp, TCNNNeRFSmall):
            outputs_flat = mlp(embedded)
        else:
            if netchunk is None:
                outputs_flat = mlp(embedded)
            else:
                outputs_flat = torch.cat(
                    [mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)],
                    0
                )

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 3]) * raw_noise_std
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise, device=raw.device)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        if not self.training and self.args.render_rmnearplane > 0:
            mask = z_vals[:, 1:] > self.args.render_rmnearplane / 128
            density = mask.type_as(density) * density

        alpha = -torch.exp(-density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device),
                       -alpha + (1. + 1e-10)], -1),
            -1
        )[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        entropy = Categorical(
            probs=torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True) + 1e-6], dim=-1)
        ).entropy()
        sparsity_loss = entropy

        return rgb_map, density, acc_map, weights, depth_map, sparsity_loss


    def _init_occ_grid(self, device):
        if self.occ_grid_res <= 0 or not hasattr(self.args, "bounding_box"):
            return
        if self.occ_grid is None or self.occ_grid.device != device:
            self.occ_grid = torch.zeros(
                (self.occ_grid_res, self.occ_grid_res, self.occ_grid_res),
                device=device, dtype=torch.float32
            )

    def _points_to_occ_idx(self, pts):
        if self.occ_grid is None or not hasattr(self.args, "bounding_box"):
            return None, None
        box_min, box_max = self.args.bounding_box
        box_min = box_min.to(device=pts.device, dtype=pts.dtype)
        box_max = box_max.to(device=pts.device, dtype=pts.dtype)
        denom = (box_max - box_min).clamp_min(1e-6)
        norm = (pts - box_min) / denom
        inside = ((norm >= 0.0) & (norm <= 1.0)).all(dim=-1)
        norm = norm.clamp(0.0, 1.0 - 1e-6)
        idx = (norm * self.occ_grid_res).long().clamp(0, self.occ_grid_res - 1)
        return idx, inside

    def _make_occ_mask(self, pts):
        if self.occ_grid_res <= 0:
            return None
        self._init_occ_grid(pts.device)
        if self.occ_grid is None:
            return None

        if self.occ_step < self.occ_warmup_steps or not self.occ_enabled:
            keep = torch.ones(pts.shape[:-1], dtype=torch.bool, device=pts.device)
        else:
            idx, inside = self._points_to_occ_idx(pts)
            if idx is None:
                return None
            occ_vals = self.occ_grid[idx[..., 0], idx[..., 1], idx[..., 2]]
            keep = (occ_vals > self.occ_threshold) & inside

        anchor_interval = max(int(self.occ_anchor_interval), 1)
        keep[..., ::anchor_interval] = True
        keep[..., 0] = True
        keep[..., -1] = True
        return keep

    def _update_occ_grid(self, pts, density):
        if self.occ_grid_res <= 0:
            return
        self._init_occ_grid(pts.device)
        if self.occ_grid is None:
            return
        idx, inside = self._points_to_occ_idx(pts)
        if idx is None:
            return

        occ = (density.detach() > self.occ_density_threshold) & inside
        if not occ.any():
            return

        flat_idx = idx[occ]
        values = torch.ones(flat_idx.shape[0], device=pts.device, dtype=torch.float32)
        new_grid = torch.zeros_like(self.occ_grid)
        new_grid.index_put_(
            (flat_idx[:, 0], flat_idx[:, 1], flat_idx[:, 2]),
            values,
            accumulate=True
        )
        new_grid.clamp_(0.0, 1.0)
        self.occ_grid.mul_(self.occ_decay).add_(new_grid * (1.0 - self.occ_decay))
        self.occ_enabled = True

    def _mlpforward_pruned(self, pts, viewdirs, mlp, keep_mask, netchunk=1024 * 64):
        if keep_mask is None or bool(keep_mask.all()):
            return self.mlpforward(pts, viewdirs, mlp, netchunk=netchunk)

        raw = torch.zeros((*pts.shape[:-1], 4), device=pts.device, dtype=pts.dtype)
        pts_kept = pts[keep_mask]
        if pts_kept.numel() == 0:
            return raw

        if viewdirs is not None:
            vd = viewdirs[:, None, :].expand(*pts.shape[:-1], viewdirs.shape[-1])
            vd_kept = vd[keep_mask]
            raw_kept = self.mlpforward(
                pts_kept[:, None, :], vd_kept, mlp, netchunk=netchunk
            )[:, 0, :]
        else:
            raw_kept = self.mlpforward(
                pts_kept[:, None, :], None, mlp, netchunk=netchunk
            )[:, 0, :]
        
        raw_kept = raw_kept.to(raw.dtype)
        raw[keep_mask] = raw_kept
        return raw

    def render_rays(self, ray_batch, N_samples, retraw=False, lindisp=False, perturb=0.,
                    N_importance=0, white_bkgd=False, raw_noise_std=0., pytest=False):
        self.occ_step += 1

        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * t_vals
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=rays_o.device).type_as(rays_o)

            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand, device=rays_o.device)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        keep_mask = self._make_occ_mask(pts)
        raw = self._mlpforward_pruned(pts, viewdirs, self.mlp_coarse, keep_mask)
        rgb_map, density_map, acc_map, weights, depth_map, sparsity_loss = self.raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

        if self.training and self.occ_grid_res > 0 and (self.occ_step % max(int(self.occ_update_every), 1) == 0):
            self._update_occ_grid(pts[:, :-1, :], density_map)

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0, sparsity_loss_0 =                 rgb_map, depth_map, acc_map, density_map, sparsity_loss

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance,
                                   det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            fine_keep_mask = self._make_occ_mask(pts)
            raw = self._mlpforward_pruned(pts, viewdirs, mlp, fine_keep_mask)

            rgb_map, density_map, acc_map, weights, depth_map, sparsity_loss = self.raw2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
            )

            if self.training and self.occ_grid_res > 0 and (self.occ_step % max(int(self.occ_update_every), 1) == 0):
                self._update_occ_grid(pts[:, :-1, :], density_map)

        ret = {
            'rgb_map': rgb_map,
            'depth_map': depth_map,
            'acc_map': acc_map,
            'density_map': density_map,
            'sparsity_loss': sparsity_loss
        }
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
            ret['sparsity_loss0'] = sparsity_loss_0

        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")

        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_info=None, poses=None, **kwargs):
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"

            force_baseline = kwargs.pop("force_naive", True)
            if self.kernelsnet is not None and not force_baseline:
                if self.kernelsnet.require_depth:
                    with torch.no_grad():
                        rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                        rays_info["ray_depth"] = depth[:, None]

                new_rays, weight, align_loss = self.kernelsnet(H, W, K, rays, rays_info)
                ray_num, pt_num = new_rays.shape[:2]

                rgb, depth, acc, extras = self.render(H, W, K, chunk, new_rays.reshape(-1, 3, 2), **kwargs)
                rgb_pts = rgb.reshape(ray_num, pt_num, 3)

                rgb = torch.sum(rgb_pts * weight[..., None], dim=1)
                rgb = self.tonemapping(rgb)

                rgb0 = None
                if kwargs['N_importance'] > 0:
                    rgb0_pts = extras['rgb0'].reshape(ray_num, pt_num, 3)
                    rgb0 = torch.sum(rgb0_pts * weight[..., None], dim=1)
                    rgb0 = self.tonemapping(rgb0)

                other_loss = {}
                if align_loss is not None:
                    other_loss["align"] = align_loss.reshape(1, 1)

                ngp_loss = {'sparsity_loss': extras['sparsity_loss']}
                if kwargs['N_importance'] > 0:
                    ngp_loss['sparsity_loss0'] = extras['sparsity_loss0']

                return rgb, rgb0, other_loss, ngp_loss
            else:
                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                ngp_loss = {'sparsity_loss': extras['sparsity_loss']}
                if kwargs['N_importance'] > 0:
                    ngp_loss['sparsity_loss0'] = extras['sparsity_loss0']

                rgb0 = self.tonemapping(extras['rgb0']) if ('rgb0' in extras) else None
                return self.tonemapping(rgb), rgb0, {}, ngp_loss

        else:
            assert poses is not None, "Please specify poses when in the eval model"

            if "render_point" in kwargs.keys():
                rgbs, depths, weights = self.render_subpath(H, W, K, chunk, poses, **kwargs)
                depths = weights * 2

            elif self.kernelsnet is not None and "images_indices" in kwargs:
                rgbs, depths = self.render_path_kernel(
                    H, W, K, chunk,
                    poses,
                    kwargs["render_kwargs"],
                    kwargs["images_indices"],
                    kwargs.get("render_factor", 0)
                )

            else:
                rgbs, depths = self.render_path(H, W, K, chunk, poses, **kwargs)

            return self.tonemapping(rgbs), depths

    def render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            viewdirs = rays_d
            if c2w_staticcam is not None:
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        else:
            viewdirs = None

        sh = rays_d.shape
        if ndc:
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path(self, H, W, K, chunk, render_poses, render_kwargs, render_factor=0):
        if render_factor != 0:
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            rgb, depth, acc, extras = self.render(
                H, W, K, chunk=chunk, rays=rays, c2w=c2w[:3, :4], **render_kwargs
            )

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        return rgbs, depths

    def render_path_kernel(self, H, W, K, chunk, render_poses, render_kwargs, images_indices, render_factor=0):
        """
        Render poses using the deblur kernel, exactly like the training-time fusion.
        """
        if render_factor != 0:
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        t = time.time()

        rayx, rayy = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=render_poses.device),
            torch.linspace(0, H - 1, H, device=render_poses.device),
            indexing='ij'
        )
        rayx = rayx.t().reshape(-1, 1) + HALF_PIX
        rayy = rayy.t().reshape(-1, 1) + HALF_PIX

        for imgidx, c2w in zip(images_indices, render_poses):
            i = int(imgidx.item())
            print(i, time.time() - t)
            t = time.time()

            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1).reshape(H * W, 3, 2)

            rays_info = {}

            if self.kernelsnet.require_depth:
                with torch.no_grad():
                    _, depth, _, _ = self.render(H, W, K, chunk, rays, **render_kwargs)
                    rays_info["ray_depth"] = depth[..., None]

            i = i if i < self.kernelsnet.num_img else 1
            rays_info["images_idx"] = torch.ones_like(rays[:, 0:1, 0]).long() * i
            rays_info["rays_x"] = rayx
            rays_info["rays_y"] = rayy

            new_rays, weight, _ = self.kernelsnet(H, W, K, rays, rays_info)
            ray_num, pt_num = new_rays.shape[:2]

            rgb, depth, acc, extras = self.render(
                H, W, K,
                chunk=chunk,
                rays=new_rays.reshape(-1, 3, 2),
                c2w=c2w[:3, :4],
                **render_kwargs
            )

            rgb_pts = rgb.reshape(ray_num, pt_num, 3)
            rgb = torch.sum(rgb_pts * weight[..., None], dim=1)

            depth_pts = depth.reshape(ray_num, pt_num)
            depth = torch.sum(depth_pts * weight, dim=1)

            rgbs.append(rgb.reshape(H, W, 3))
            depths.append(depth.reshape(H, W))

            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        return rgbs, depths

    def render_subpath(self, H, W, K, chunk, render_poses, render_point, images_indices, render_kwargs,
                       render_factor=0):
        if render_factor != 0:
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []
        weights = []

        t = time.time()

        rayx, rayy = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=render_poses.device),
            torch.linspace(0, H - 1, H, device=render_poses.device),
            indexing='ij'
        )
        rayx = rayx.t().reshape(-1, 1) + HALF_PIX
        rayy = rayy.t().reshape(-1, 1) + HALF_PIX

        for imgidx, c2w in zip(images_indices, render_poses):
            i = int(imgidx.item())
            print(i, time.time() - t)
            t = time.time()

            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1).reshape(H * W, 3, 2)

            rays_info = {}
            if self.kernelsnet.require_depth:
                with torch.no_grad():
                    rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **render_kwargs)
                    rays_info["ray_depth"] = depth[..., None]

            i = i if i < self.kernelsnet.num_img else 1
            rays_info["images_idx"] = torch.ones_like(rays[:, 0:1, 0]).long() * i
            rays_info["rays_x"] = rayx
            rays_info["rays_y"] = rayy

            new_rays, weight, _ = self.kernelsnet(H, W, K, rays, rays_info)

            new_rays = new_rays[:, render_point]
            weight = weight[:, render_point]
            rgb, depth, acc, extras = self.render(
                H, W, K, chunk=chunk, rays=new_rays.reshape(-1, 3, 2),
                c2w=c2w[:3, :4], **render_kwargs
            )

            rgbs.append(rgb.reshape(H, W, 3))
            depths.append(depth.reshape(H, W))
            weights.append(weight.reshape(H, W))

            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        weights = torch.stack(weights, 0)

        return rgbs, depths, weights