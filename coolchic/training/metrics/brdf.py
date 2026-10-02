import math

import torch
import torch.nn.functional as f
from coolchic.utils import color, hammersley

# The two resources below were used as reference when implementing BRDF and PBR
# https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation
# https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/bec106e53da4a6a398aa3205f0f96563519a657e/source/Renderer/shaders/brdf.glsl

def get_orthonormal_basis_vectors(normal):
    pointing_down = normal[..., [2]] < 0.0
    if torch.any(pointing_down):
        raise ValueError("Expected all normals to be pointing up")

    a = 1.0 / (1.0 + torch.clamp(normal[..., 2], min=0.0))
    b = -normal[..., 0] * normal[..., 1] * a
    t1 = torch.stack(
        (1.0 - normal[..., 0] * normal[..., 0] * a, b, -normal[..., 0]), dim=-1
    )
    t2 = torch.stack(
        (b, 1.0 - normal[..., 1] * normal[..., 1] * a, -normal[..., 1]), dim=-1
    )

    return (t1, t2)


def build_orhtonormal_basis(normal):
    (t1, t2) = get_orthonormal_basis_vectors(normal)

    return torch.stack((t1, t2, normal), dim=-1)


def diffuse_pdf(n_dot_l):
    pdf = n_dot_l / torch.pi

    return torch.where(n_dot_l < 0, 0, pdf)


def specular_ggx_ndf(alpha_rough, tangent_half_vector):
    a2 = alpha_rough * alpha_rough
    cos2theta = tangent_half_vector[..., [2]] * tangent_half_vector[..., [2]]
    sin2theta = torch.clamp(1.0 - cos2theta, min=0.0)
    tan2theta = sin2theta / cos2theta
    sq_part_denom = a2 + tan2theta
    ndf = a2 / (torch.pi * cos2theta * cos2theta * (sq_part_denom * sq_part_denom))

    return torch.where(
        torch.logical_or(
            torch.logical_or(tangent_half_vector[..., [2]] <= 0.0, alpha_rough <= 0.0),
            torch.isnan(ndf),
        ),
        0.0,
        ndf,
    )


def specular_ggx_vndf_pdf(alpha_rough, tangent_eye_dir, tangent_half_vector):
    ndf = specular_ggx_ndf(alpha_rough, tangent_half_vector)
    aO = alpha_rough * tangent_eye_dir[..., 0:2]
    len2 = aO[..., [0]] * aO[..., [0]] + aO[..., [1]] * aO[..., [1]]
    t = torch.sqrt(len2 + tangent_eye_dir[..., [2]] * tangent_eye_dir[..., [2]])

    a = torch.clamp(alpha_rough, min=0.0, max=1.0)
    s = 1.0 + torch.sqrt(
        torch.sum(
            tangent_eye_dir[..., 0:2] * tangent_eye_dir[..., 0:2],
            dim=-1,
            keepdim=True,
        )
    )
    a2 = a * a
    s2 = s * s
    k_denom = s2 + a2 * tangent_eye_dir[..., [2]] * tangent_eye_dir[..., [2]]
    k = torch.where(k_denom <= 0.0, 0.0, (1.0 - a2) * s2 / k_denom)

    result_denom_0 = 2.0 * (k * tangent_eye_dir[..., [2]] + t)
    result_denom_1 = 2.0 * len2

    return torch.where(
        tangent_eye_dir[..., [2]] > 0.0,
        torch.where(result_denom_0 <= 0.0, 0.0, ndf / result_denom_0),
        torch.where(
            result_denom_1 <= 0.0,
            0.0,
            ndf * (t - tangent_eye_dir[..., [2]]) / result_denom_1,
        ),
    )


def reflect_vector(vector, normal):
    """
    Both vectors are expected to be normalized
    """

    if vector.shape[-1] != 3 or normal.shape[-1] != 3:
        raise ValueError(
            f"Expected vector and normal to have 3 channels in last dimension but got {vector.shape[-1]} and {normal.shape[-1]} channels respectively"
        )

    reflected_vector = (
        vector - (2 * torch.sum(normal * vector, dim=-1, keepdim=True)) * normal
    )

    return reflected_vector


def sample_specular_ggx_vndf(tangent_eye_dir, alpha_rough, u, v):
    if len(tangent_eye_dir.shape) != len(alpha_rough.shape):
        raise ValueError(
            f"'tangent_eye_dir' and 'alpha_rough' should have same num dims but got {len(tangent_eye_dir.shape)} and {len(alpha_rough.shape)} dims"
        )

    if len(tangent_eye_dir.shape) != len(u.shape):
        raise ValueError(
            f"'tangent_eye_dir' and 'u' should have same num dims but got {len(tangent_eye_dir.shape)} and {len(u.shape)} dims"
        )

    if len(tangent_eye_dir.shape) != len(v.shape):
        raise ValueError(
            f"'tangent_eye_dir' and 'v' should have same num dims but got {len(tangent_eye_dir.shape)} and {len(v.shape)} dims"
        )

    if tangent_eye_dir.shape[-1] != 3:
        raise ValueError(
            f"Expected vector to have 3 channels in last dimension but got {tangent_eye_dir.shape[-1]} channels"
        )

    if alpha_rough.shape[-1] != 1:
        raise ValueError(
            f"Expected vector to have 1 channel in last dimension but got {alpha_rough.shape[-1]} channels"
        )

    normalized_eye_dir = f.normalize(
        torch.concat(
            (
                tangent_eye_dir[..., [0]] * alpha_rough,
                tangent_eye_dir[..., [1]] * alpha_rough,
                tangent_eye_dir[..., [2]],
            ),
            dim=-1,
        )
    )

    phi = 2 * torch.pi * v
    a = torch.clamp(alpha_rough, min=0.0, max=1.0)
    s = 1.0 + torch.sqrt(
        torch.pow(tangent_eye_dir[..., [0]], 2)
        + torch.pow(tangent_eye_dir[..., [1]], 2)
    )
    a2 = a * a
    s2 = s * s

    k_denom = s2 + a2 * tangent_eye_dir[..., [2]] * tangent_eye_dir[..., [2]]
    k = torch.where(k_denom <= 0.0, 0.0, (1.0 - a2) * s2 / k_denom)
    b = torch.where(
        tangent_eye_dir[..., [2]] > 0,
        k * normalized_eye_dir[..., [2]],
        normalized_eye_dir[..., [2]],
    )

    z = ((1.0 - u) * (1.0 + b)) - b
    sin_theta = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0, max=1.0))
    light_dir = torch.concat(
        (sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), z), dim=-1
    )
    half_vector = normalized_eye_dir + light_dir

    tangent_half_vector = f.normalize(
        torch.concat(
            (
                half_vector[..., [0]] * alpha_rough,
                half_vector[..., [1]] * alpha_rough,
                torch.clamp(half_vector[..., [2]], min=0.0),
            ),
            dim=-1,
        )
    )

    return tangent_half_vector


def sample_specular_lobe(
    shape, N, tangent_eye_dir, rough, *, generator=None, device=None
):
    rand_indices = torch.rand(shape, generator=generator, device=device)
    rand_indices = torch.floor(rand_indices * N)

    u, v = hammersley.hammersley2d_torch(rand_indices, N)
    tangent_half_vector = sample_specular_ggx_vndf(
        tangent_eye_dir, torch.pow(rough, 2), u, v
    )
    tangent_light_dir = reflect_vector(-tangent_eye_dir, tangent_half_vector)

    return tangent_light_dir

def lambertian_brdf(diffuse: torch.Tensor):
    return diffuse / torch.pi


def V_GGX(n_dot_l, n_dot_v, alpha_rough_sq):
    ggxv = n_dot_l * torch.sqrt(
        torch.clamp(
            n_dot_v * n_dot_v * (1.0 - alpha_rough_sq) + alpha_rough_sq, min=1e-8
        )
    )
    ggxl = n_dot_v * torch.sqrt(
        torch.clamp(
            n_dot_l * n_dot_l * (1.0 - alpha_rough_sq) + alpha_rough_sq, min=1e-8
        )
    )

    ggx = ggxv + ggxl

    # masked_ggx is used to avoid nans in the backward call
    masked_ggx = torch.where(ggx > 0.0, ggx, torch.ones_like(ggx))
    return torch.where(ggx > 0.0, 0.5 / masked_ggx, torch.zeros_like(masked_ggx))


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


def pbr_log_tone_mapping(color: torch.Tensor):
    color = torch.where(color < 0.0, 0.0, color)

    log_mapped = (torch.log(color + 0.01) - math.log(0.01)) / (
        math.log(1.01) - math.log(0.01)
    )

    return log_mapped


def calc_pbr(
    light_dir,
    eye_dir,
    diffuse,
    roughness,
    metal,
    normal=None,
    pdf=None,
    apply_shading=True,
    epsilon=1e-8,
):
    """
    If normals are None the light and eye directions are expected to be in tangent space.
    Normals are expected to have a range of [-1, 1].
    apply_shading determines if the results are multiplied by N dot L.
    """

    # TODO (Alex): This is temporary as this should be fixed at load time in a different PR.
    diffuse = color.srgb_to_linear_torch(diffuse)

    half_vector = torch.nn.functional.normalize(light_dir + eye_dir, dim=-1)

    alpha_roughness = roughness * roughness

    if normal is None:
        n_dot_l = light_dir[..., [2]]
        n_dot_v = eye_dir[..., [2]]
        n_dot_h = half_vector[..., [2]]
    else:
        n_dot_l = torch.sum(normal * light_dir, dim=-1, keepdim=True)
        n_dot_v = torch.sum(normal * eye_dir, dim=-1, keepdim=True)
        n_dot_h = torch.sum(normal * half_vector, dim=-1, keepdim=True)

    v_dot_h = torch.sum(eye_dir * half_vector, dim=-1, keepdim=True)

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

    final_brdf = (1.0 - metal) * dielectric_brdf + metal_brdf * metal

    if pdf is not None:
        final_brdf = final_brdf / torch.clamp(pdf, min=epsilon)

    return final_brdf


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
