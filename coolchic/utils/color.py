import torch


def linear_to_srgb_torch(linear_srgb: torch.Tensor) -> torch.Tensor:
    limit = 0.0031308
    return torch.where(
        linear_srgb > limit,
        1.055 * torch.pow(torch.clamp(linear_srgb, min=limit), (1.0 / 2.4)) - 0.055,
        12.92 * linear_srgb,
    )


def srgb_to_linear_torch(srgb: torch.Tensor) -> torch.Tensor:
    limit = 0.04045
    return torch.where(
        srgb > limit,
        torch.pow((torch.clamp(srgb, min=limit) + 0.055) / 1.055, 2.4),
        srgb / 12.92,
    )