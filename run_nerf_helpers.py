import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hash_encoding import HashEmbedder, SHEncoder

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    tcnn = None
    HAS_TCNN = False

print("run_nerf_helpers HAS_TCNN =", HAS_TCNN)

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.0], device=x.device))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

HALF_PIX = 0.5


class ToneMapping(nn.Module):
    def __init__(self, map_type: str):
        super(ToneMapping, self).__init__()
        assert map_type in ['none', 'gamma', 'learn', 'ycbcr']
        self.map_type = map_type
        if map_type == 'learn':
            self.linear = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )

    def forward(self, x):
        if self.map_type == 'none':
            return x
        elif self.map_type == 'learn':
            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            res_x = self.linear(x_in) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            return x_out.reshape(ori_shape)
        elif self.map_type == 'gamma':
            return x ** (1. / 2.2)
        else:
            raise RuntimeError("map_type not recognized")


# -------------------------
# Positional encoding
# -------------------------
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0

        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        n_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=n_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=n_freqs)

        for _freq in self.freq_bands:
            for _ in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))

        return torch.cat(outputs, -1)


# -------------------------
# tiny-cuda-nn Hash encoder
# -------------------------
class TCNNHashEmbedder(nn.Module):
    def __init__(
        self,
        bounding_box,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
        n_levels=16,
        n_features_per_level=2,
    ):
        super().__init__()
        self.bounding_box = bounding_box
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level

        per_level_scale = np.exp(
            (np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1)
        )

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.out_dim = self.encoder.n_output_dims

    def forward(self, x):
        box_min, box_max = self.bounding_box
        box_min = box_min.to(device=x.device, dtype=x.dtype)
        box_max = box_max.to(device=x.device, dtype=x.dtype)

        x = (x - box_min) / (box_max - box_min + 1e-8)
        x = torch.clamp(x, 0.0, 1.0)
        return self.encoder(x.contiguous())


def get_embedder(multires, args, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), input_dim

    elif i == 0:
        embedder_obj = Embedder(
            include_input=True,
            input_dims=input_dim,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

    elif i == 1:
        if HAS_TCNN:
            print("Using tiny-cuda-nn HashGrid encoder")
            embedder_obj = TCNNHashEmbedder(
                bounding_box=args.bounding_box,
                log2_hashmap_size=args.log2_hashmap_size,
                finest_resolution=args.finest_res,
                base_resolution=16,
                n_levels=16,
                n_features_per_level=2,
            )
        else:
            embedder_obj = HashEmbedder(
                bounding_box=args.bounding_box,
                log2_hashmap_size=args.log2_hashmap_size,
                finest_resolution=args.finest_res,
            )

    elif i == 2:
        embedder_obj = SHEncoder()

    else:
        raise ValueError(f"Unknown embedder type {i}")

    return embedder_obj, embedder_obj.out_dim


# -------------------------
# Original NeRF
# -------------------------
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
             for i in range(D - 1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, _ in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# -------------------------
# Small NeRF (PyTorch fallback)
# -------------------------
class NeRFSmall(nn.Module):
    def __init__(
        self,
        num_layers=3,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=4,
        hidden_dim_color=64,
        input_ch=3,
        input_ch_views=3,
    ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        sigma_net = []
        for l in range(num_layers):
            in_dim = self.input_ch if l == 0 else hidden_dim
            out_dim = 1 + self.geo_feat_dim if l == num_layers - 1 else hidden_dim
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.sigma_net = nn.ModuleList(sigma_net)

        color_net = []
        for l in range(num_layers_color):
            in_dim = self.input_ch_views + self.geo_feat_dim if l == 0 else hidden_dim_color
            out_dim = 3 if l == num_layers_color - 1 else hidden_dim_color
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
        self.color_net = nn.ModuleList(color_net)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        h = torch.cat([input_views, geo_feat], dim=-1)

        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        color = h
        return torch.cat([color, sigma.unsqueeze(dim=-1)], -1)


# -------------------------
# Small NeRF (tiny-cuda-nn)
# -------------------------
class TCNNNeRFSmall(nn.Module):
    def __init__(
        self,
        input_ch,
        input_ch_views,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers=2,
        num_layers_color=3,
    ):
        super().__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.geo_feat_dim = geo_feat_dim

        sigma_hidden_layers = max(num_layers - 1, 1)
        color_hidden_layers = max(num_layers_color - 1, 1)

        self.sigma_net = tcnn.Network(
            n_input_dims=input_ch,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": sigma_hidden_layers,
            },
        )

        self.color_net = tcnn.Network(
            n_input_dims=input_ch_views + geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": color_hidden_layers,
            },
        )

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )

        sigma_geo = self.sigma_net(input_pts.contiguous())
        sigma = sigma_geo[..., :1]
        geo_feat = sigma_geo[..., 1:]

        h = torch.cat([input_views, geo_feat], dim=-1)
        rgb = self.color_net(h.contiguous())

        return torch.cat([rgb, sigma], dim=-1)


# -------------------------
# Ray helpers
# -------------------------
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device),
        indexing='ij'
    )
    i = i.t()
    j = j.t()
    dirs = torch.stack([
        (i + (HALF_PIX - K[0][2])) / K[0][0],
        -(j + (HALF_PIX - K[1][2])) / K[1][1],
        -torch.ones_like(i)
    ], -1)

    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([
        (i + (HALF_PIX - K[0][2])) / K[0][0],
        -(j + (HALF_PIX - K[1][2])) / K[1][1],
        -np.ones_like(i)
    ], -1)

    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d


# -------------------------
# Hierarchical sampling
# -------------------------
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).to(bins.device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


# -------------------------
# Checkpoint loading
# -------------------------
def smart_load_state_dict(model: nn.Module, state_dict: dict):
    if "network_fn_state_dict" in state_dict:
        state_dict_fn = {k.lstrip("module."): v for k, v in state_dict["network_fn_state_dict"].items()}
        state_dict_fn = {"mlp_coarse." + k: v for k, v in state_dict_fn.items()}

        state_dict_fine = {k.lstrip("module."): v for k, v in state_dict["network_fine_state_dict"].items()}
        state_dict_fine = {"mlp_fine." + k: v for k, v in state_dict_fine.items()}
        state_dict_fn.update(state_dict_fine)
        state_dict = state_dict_fn

    elif "network_state_dict" in state_dict:
        raw_state = state_dict["network_state_dict"]
        if all(k.startswith("module.") for k in raw_state.keys()):
            state_dict = {k[7:]: v for k, v in raw_state.items()}
        else:
            state_dict = raw_state
    else:
        state_dict = state_dict

    if isinstance(model, nn.DataParallel):
        if not all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {"module." + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)