from math import exp, log, floor
import pdb
import torch

from utils import hash


def total_variation_loss(embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16):
    b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1))
    resolution = torch.tensor(floor(min_resolution * b ** level))

    min_cube_size = min_resolution - 1
    max_cube_size = 50
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = torch.floor(torch.clip(resolution / 10.0, min_cube_size, max_cube_size)).int()

    min_vertex = torch.randint(0, resolution - cube_size, (3,))
    idx = min_vertex + torch.stack([torch.arange(cube_size + 1) for _ in range(3)], dim=-1)
    cube_indices = torch.stack(torch.meshgrid(idx[:, 0], idx[:, 1], idx[:, 2], indexing='ij'), dim=-1)

    hashed_indices = hash(cube_indices, log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)

    tv_x = torch.pow(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2).sum()
    tv_y = torch.pow(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2).sum()
    tv_z = torch.pow(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2).sum()

    return (tv_x + tv_y + tv_z) / cube_size


def sigma_sparsity_loss(sigmas):
    return torch.log(1.0 + 2 * sigmas ** 2).sum(dim=-1)
