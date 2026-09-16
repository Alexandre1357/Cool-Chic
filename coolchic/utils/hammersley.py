import math
import torch

# NOTE (Alex): Sources used as reference
# https://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
# https://learnopengl.com/PBR/IBL/Specular-IBL


def van_der_corput(n, base):
    inv_base = 1.0 / base
    denom = 1.0
    result = 0.0

    for _ in range(32):
        if n > 0:
            denom = n % 2
            result += denom * inv_base
            inv_base = inv_base / 2
            n = math.floor(n / 2)

    return result


def van_der_corput_torch(n, base):
    inv_base = 1.0 / base
    denom = torch.ones_like(n)
    result = torch.zeros_like(n)

    for _ in range(32):
        condition = n > 0
        denom = torch.where(condition, n % 2, denom)
        result = torch.where(condition, result + denom * inv_base, result)
        inv_base = torch.where(condition, inv_base / 2, inv_base)
        n = torch.where(condition, torch.floor(n / 2), n)

    return result


def hammersley2d(i, N):
    return i / N, van_der_corput(i, 2)


def hammersley2d_torch(i, N):
    return i / N, van_der_corput_torch(i, 2)


def sample_hemisphere_uniform(u, v):
    phi = v * 2.0 * math.pi
    cos_theta = 1.0 - u
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
    return math.cos(phi) * sin_theta, math.sin(phi) * sin_theta, cos_theta


def sample_hemisphere_cos(u, v):
    phi = v * 2.0 * math.pi
    cos_theta = math.sqrt(1.0 - u)
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
    return math.cos(phi) * sin_theta, math.sin(phi) * sin_theta, cos_theta


def sample_hemisphere_torch(u, v, uniform=True):
    phi = v * 2.0 * torch.pi
    cos_theta = 1.0 - u
    if not uniform:
        cos_theta = torch.sqrt(cos_theta)
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
    return torch.cos(phi) * sin_theta, torch.sin(phi) * sin_theta, cos_theta


def hemisphere_samples_torch(N, uniform=True) -> torch.Tensor:
    indices = torch.linspace(0, N - 1, N)

    x, y = hammersley2d_torch(indices, N)
    x, y, z = sample_hemisphere_torch(x, y, uniform)
    samples_3d = torch.stack((x, y, z), dim=-1)

    return samples_3d


def rand_sample_hemisphere_torch(
    shape, N, *, uniform=True, generator=None, device=None
):
    rand_indices = torch.rand(shape, generator=generator, device=device)
    rand_indices = torch.floor(rand_indices * N)

    x_samples, y_samples = hammersley2d_torch(rand_indices, N)
    x_samples, y_samples, z_samples = sample_hemisphere_torch(
        x_samples, y_samples, uniform
    )

    samples_3d = torch.stack((x_samples, y_samples, z_samples), dim=-1)

    return samples_3d
