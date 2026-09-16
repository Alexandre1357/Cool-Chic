import math

import torch
import torch.nn.functional as f
from coolchic.utils import color

# The two resources below were used as reference when implementing BRDF and PBR
# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
# https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/bec106e53da4a6a398aa3205f0f96563519a657e/source/Renderer/shaders/brdf.glsl


def lambertian_brdf(diffuse: torch.Tensor):
    return diffuse / torch.pi


def V_GGX_util(n_dot_l, n_dot_v, alpha_rough_sq):
    ggxv = n_dot_l * torch.sqrt(
        n_dot_v * n_dot_v * (1.0 - alpha_rough_sq) + alpha_rough_sq
    )
    ggxl = n_dot_v * torch.sqrt(
        n_dot_l * n_dot_l * (1.0 - alpha_rough_sq) + alpha_rough_sq
    )

    return ggxv + ggxl


def V_GGX(n_dot_l, n_dot_v, alpha_rough_sq):
    ggxv = n_dot_l * torch.sqrt(
        n_dot_v * n_dot_v * (1.0 - alpha_rough_sq) + alpha_rough_sq
    )
    ggxl = n_dot_v * torch.sqrt(
        n_dot_l * n_dot_l * (1.0 - alpha_rough_sq) + alpha_rough_sq
    )

    ggx = ggxv + ggxl

    # masked_ggx is used to avoid nans in the backward call
    masked_ggx = torch.where(ggx > 0.0, ggx, torch.ones_like(ggx))
    return torch.where(ggx > 0.0, 0.5 / masked_ggx, torch.zeros_like(masked_ggx))


def D_GGX_util(n_dot_h, alpha_rough_sq):
    return (n_dot_h * n_dot_h) * (alpha_rough_sq - 1.0) + 1.0


def D_GGX(n_dot_h, alpha_rough_sq, epsilon=1e-7):
    prod = (n_dot_h * n_dot_h) * (alpha_rough_sq - 1.0) + 1.0

    masked_prod = torch.where(prod <= 0.0, torch.full_like(prod, epsilon), prod)
    return torch.where(
        alpha_rough_sq <= 0.0,
        1.0,
        alpha_rough_sq / (torch.pi * masked_prod * masked_prod),
    )


def specular_GGX(alpha_rough, n_dot_l, n_dot_v, n_dot_h):
    alpha_rough_sq = alpha_rough * alpha_rough
    visibility = V_GGX(n_dot_l, n_dot_v, alpha_rough_sq)
    distribution = D_GGX(n_dot_h, alpha_rough_sq)

    return visibility * distribution


def F_Shlick(f0, f90, v_dot_h):
    return f0 + (f90 - f0) * torch.pow(
        torch.max(
            torch.min(1.0 - v_dot_h, torch.ones_like(v_dot_h)),
            torch.zeros_like(v_dot_h),
        ),
        5.0,
    )


def spherical_to_cartesian(phi, theta):
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)

    return (x, y, z)


def spherical_to_cartesian_torch(phi, theta):
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    return torch.stack((x, y, z), dim=-1)


def pbr_neutral_tone_mapping(color: torch.Tensor, color_dim: int, epsilon=1e-7):
    if color.shape[color_dim] != 3:
        raise ValueError(f"dimension {color_dim} should have 3 channels but got {color.shape[color_dim]}")

    start_compression = 0.8 - 0.04
    desaturation = 0.15

    x = torch.min(color, dim=color_dim, keepdim=True).values
    offset = torch.where(x < 0.08, x - 6.25 * x * x, 0.04)
    offset_color = color - offset

    peak = torch.max(offset_color, dim=color_dim, keepdim=True).values

    d = 1 - start_compression
    new_peak = 1 - d * d / torch.clamp(peak + d - start_compression, min=epsilon)
    norm_peak_color = offset_color * (new_peak / torch.clamp(peak, min=epsilon))

    g = 1 - 1 / torch.clamp(desaturation * (peak - new_peak) + 1, min=epsilon)
    mixed_color = (
        norm_peak_color * (1.0 - g) + (new_peak * torch.ones_like(norm_peak_color)) * g
    )

    return torch.where(peak < start_compression, offset_color, mixed_color)


def calc_pbr(
    light_dir,
    eye_dir,
    diffuse,
    normal,
    roughness,
    metal,
    apply_shading=True,
    channel_dim=-1,
    epsilon=1e-8,
) -> torch.Tensor:
    """
    Normals are expected to have a range of [-1, 1].
    apply_shading determines if the results are multiplied by N dot L.
    """

    if light_dir.shape[channel_dim] != 3 or eye_dir.shape[channel_dim] != 3:
        raise ValueError(
            f"Expected eye and light directions to have 3 channels for dim {channel_dim} but got {eye_dir.shape[channel_dim]} and {light_dir.shape[channel_dim]} channels respectively"
        )

    if len(diffuse.shape) == 3:
        diffuse = diffuse.unsqueeze(0).unsqueeze(0)
    elif len(diffuse.shape) == 4:
        diffuse = diffuse.unsqueeze(0)
    elif len(diffuse.shape) != 5:
        raise ValueError(
            f"Expected 5 dimension for the diffuse texture got {len(diffuse.shape)}"
        )
    
    if diffuse.shape[channel_dim] != 3:
        raise ValueError(
            f"Expected 3 channels for for diffuse dim {channel_dim} but got {diffuse.shape[channel_dim]}"
        )

    if len(normal.shape) == 3:
        normal = normal.unsqueeze(0).unsqueeze(0)
    elif len(normal.shape) == 4:
        normal = normal.unsqueeze(0)
    elif len(normal.shape) != 5:
        raise ValueError(
            f"Expected 5 dimension for the normal texture got {len(normal.shape)}"
        )
    
    if normal.shape[channel_dim] != 3:
        raise ValueError(
            f"Expected 3 channels for for normal dim {channel_dim} but got {normal.shape[channel_dim]}"
        )

    if len(roughness.shape) == 3:
        roughness = roughness.unsqueeze(0).unsqueeze(0)
    elif len(roughness.shape) == 4:
        roughness = roughness.unsqueeze(0)
    elif len(roughness.shape) != 5:
        raise ValueError(
            f"Expected 5 dimension for the roughness texture got {len(roughness.shape)}"
        )
    
    if roughness.shape[channel_dim] != 1:
        raise ValueError(
            f"Expected 1 channels for for roughness dim {channel_dim} but got {roughness.shape[channel_dim]}"
        )

    if len(metal.shape) == 3:
        metal = metal.unsqueeze(0).unsqueeze(0)
    elif len(metal.shape) == 4:
        metal = metal.unsqueeze(0)
    elif len(metal.shape) != 5:
        raise ValueError(
            f"Expected 5 dimension for the metal texture got {len(metal.shape)}"
        )
    
    if metal.shape[channel_dim] != 1:
        raise ValueError(
            f"Expected 1 channels for for metal dim {channel_dim} but got {metal.shape[channel_dim]}"
        )

    diffuse = color.srgb_to_linear_torch(diffuse)

    half_vector = torch.nn.functional.normalize(light_dir + eye_dir, dim=channel_dim)

    alpha_roughness = roughness * roughness
    n_dot_l = torch.sum(normal * light_dir, dim=channel_dim, keepdim=True)
    n_dot_v = torch.sum(normal * eye_dir, dim=channel_dim, keepdim=True)
    n_dot_h = torch.sum(normal * half_vector, dim=channel_dim, keepdim=True)
    v_dot_h = torch.sum(eye_dir * half_vector, dim=channel_dim, keepdim=True)

    metal_fresnel = F_Shlick(diffuse, 1.0, torch.abs(v_dot_h))
    dielectric_fresnel = F_Shlick(0.04, 1.0, torch.abs(v_dot_h))

    applied_shading = torch.clamp(n_dot_l, 0.0, 1.0) if apply_shading else 1.0

    diffuse_brdf = lambertian_brdf(diffuse) * applied_shading
    specular_brdf = (
        specular_GGX(
            torch.clamp(alpha_roughness, epsilon),
            torch.clamp(n_dot_l, 0.0, 1.0),
            torch.clamp(n_dot_v, 0.0, 1.0),
            torch.clamp(n_dot_h, 0.0, 1.0),
        )
        * applied_shading
    )
    metal_brdf = metal_fresnel * specular_brdf
    dielectric_brdf = (
        1.0 - dielectric_fresnel
    ) * diffuse_brdf + specular_brdf * dielectric_fresnel

    return (1.0 - metal) * dielectric_brdf + metal_brdf * metal


def organize_textures(
    stacked_textures: torch.Tensor,
    epsilon=1e-7,
):
    diffuse = stacked_textures[:, 0:3, ...]

    normals = stacked_textures[:, 3:5, ...]
    normals = normals * 2.0 - 1.0

    z2 = (1.0 - normals.pow(2).sum(dim=1, keepdim=True)).clamp(epsilon, 1.0)
    normals = torch.cat([normals, z2.sqrt()], dim=1)
    normals = f.normalize(normals, dim=1)

    rough = stacked_textures[:, 5:6, ...]
    metal = stacked_textures[:, 6:7, ...]

    return (diffuse, normals, rough, metal)
