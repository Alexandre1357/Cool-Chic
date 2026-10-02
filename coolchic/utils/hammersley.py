import math
import torch

# NOTE (Alex): Sources used as reference
# https://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
# https://learnopengl.com/PBR/IBL/Specular-IBL

def radical_inverse_torch(bits: torch.Tensor):
    bits = bits.to(dtype=torch.int32)

    bits = torch.bitwise_or(
        torch.bitwise_left_shift(bits, 16), torch.bitwise_right_shift(bits, 16)
    )
    bits = torch.bitwise_or(
        torch.bitwise_left_shift(torch.bitwise_and(bits, 0x55555555), 1),
        torch.bitwise_right_shift(torch.bitwise_and(bits, 0xAAAAAAAA), 1),
    )
    bits = torch.bitwise_or(
        torch.bitwise_left_shift(torch.bitwise_and(bits, 0x33333333), 2),
        torch.bitwise_right_shift(torch.bitwise_and(bits, 0xCCCCCCCC), 2),
    )
    bits = torch.bitwise_or(
        torch.bitwise_left_shift(torch.bitwise_and(bits, 0x0F0F0F0F), 4),
        torch.bitwise_right_shift(torch.bitwise_and(bits, 0xF0F0F0F0), 4),
    )
    bits = torch.bitwise_or(
        torch.bitwise_left_shift(torch.bitwise_and(bits, 0x00FF00FF), 8),
        torch.bitwise_right_shift(torch.bitwise_and(bits, 0xFF00FF00), 8),
    )
    bits = bits.to(dtype=torch.uint32)

    bits = bits.float() * 2.3283064365386963e-10
    return bits

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
    return i / N, radical_inverse_torch(i)


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
