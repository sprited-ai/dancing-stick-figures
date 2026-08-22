"""Small differentiable renderer for projected skeletons.

This is an experimental building block for a future structured video model.  It
turns 2-D projected joints and per-joint depth into soft coloured capsules.  It
does not replace the dataset renderer: the purpose is to provide gradients for
video--motion consistency losses.

Coordinates are pixel coordinates by default (x in [0, W-1], y in [0, H-1]).
Set ``normalized_coordinates=True`` to use coordinates in [-1, 1].  Smaller
depth values are treated as nearer to the camera.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import torch
from torch import Tensor, nn


def _as_size(image_size: Union[int, Sequence[int]]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or (height, width)")
    return int(image_size[0]), int(image_size[1])


class SoftSkeletonRenderer(nn.Module):
    """Render skeleton bones as differentiable soft capsules.

    Args:
        parents: Parent index for every joint; roots have parent ``-1``.  Each
            non-root joint defines one bone running from its parent to itself.
        bone_colors: RGB values for either every joint ``[J, 3]`` or every
            non-root bone ``[E, 3]``. Values should be in [0, 1].
        body_radius: Capsule radius in pixels. May be scalar, ``[J]``, or
            ``[E]``.
        edge_softness: Width in pixels of the sigmoid antialiasing edge.
        depth_temperature: Soft z-buffer temperature. Lower is closer to a
            hard nearest-surface decision.
    """

    def __init__(
        self,
        parents: Union[Sequence[int], Tensor],
        bone_colors: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
        body_radius: Union[float, Tensor, Sequence[float]] = 1.5,
        image_size: Union[int, Sequence[int]] = 64,
        edge_softness: float = 0.75,
        depth_temperature: float = 0.25,
        normalized_coordinates: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        parents_t = torch.as_tensor(parents, dtype=torch.long)
        if parents_t.ndim != 1:
            raise ValueError("parents must have shape [J]")
        child = torch.nonzero(parents_t >= 0, as_tuple=False).flatten()
        if child.numel() == 0:
            raise ValueError("parents must define at least one bone")
        parent = parents_t[child]
        if torch.any(parent >= parents_t.numel()):
            raise ValueError("a parent index is outside the joint array")

        edges = torch.stack((parent, child), dim=1)
        self.register_buffer("edges", edges, persistent=True)
        self.num_joints = int(parents_t.numel())
        self.image_size = _as_size(image_size)
        self.edge_softness = float(edge_softness)
        self.depth_temperature = float(depth_temperature)
        self.normalized_coordinates = bool(normalized_coordinates)
        self.eps = float(eps)
        if self.edge_softness <= 0 or self.depth_temperature <= 0:
            raise ValueError("edge_softness and depth_temperature must be positive")

        num_bones = int(child.numel())
        if bone_colors is None:
            # A deterministic, readable fallback palette.
            hue = torch.linspace(0.15, 0.85, num_bones)
            colors = torch.stack((hue, 1.0 - hue, 0.35 + 0.3 * hue), dim=-1)
        else:
            colors = torch.as_tensor(bone_colors, dtype=torch.float32)
            if colors.shape == (self.num_joints, 3):
                colors = colors[child]
            if colors.shape != (num_bones, 3):
                raise ValueError("bone_colors must have shape [J,3] or [E,3]")
        self.register_buffer("bone_colors", colors, persistent=True)

        radii = torch.as_tensor(body_radius, dtype=torch.float32)
        if radii.ndim == 0:
            radii = radii.expand(num_bones).clone()
        elif radii.shape == (self.num_joints,):
            radii = radii[child]
        if radii.shape != (num_bones,) or torch.any(radii <= 0):
            raise ValueError("body_radius must be positive and scalar, [J], or [E]")
        self.register_buffer("bone_radii", radii, persistent=True)

    @property
    def num_bones(self) -> int:
        return int(self.edges.shape[0])

    def _pixel_joints(self, joints: Tensor) -> Tensor:
        if not self.normalized_coordinates:
            return joints
        h, w = self.image_size
        scale = joints.new_tensor(((w - 1) / 2, (h - 1) / 2))
        return (joints + 1.0) * scale

    def forward(self, joints: Tensor, joint_depth: Tensor) -> dict[str, Tensor]:
        """Render a batch of skeleton videos.

        Args:
            joints: Projected coordinates ``[B, T, J, 2]``.
            joint_depth: Camera depth ``[B, T, J]``; smaller means nearer.

        Returns:
            A dictionary containing ``rgba`` ``[B,T,4,H,W]``, ``alpha``
            ``[B,T,1,H,W]``, ``rgb`` ``[B,T,3,H,W]``, raw per-bone
            ``coverage`` and depth-aware visible ``parts`` ``[B,T,E,H,W]``.
            Visible part maps sum to alpha at each pixel.
        """
        if joints.ndim != 4 or joints.shape[-1] != 2:
            raise ValueError("joints must have shape [B,T,J,2]")
        if joint_depth.shape != joints.shape[:-1]:
            raise ValueError("joint_depth must have shape [B,T,J]")
        if joints.shape[2] != self.num_joints:
            raise ValueError(f"expected {self.num_joints} joints, got {joints.shape[2]}")
        if not (torch.is_floating_point(joints) and torch.is_floating_point(joint_depth)):
            raise TypeError("joints and joint_depth must be floating-point tensors")

        joints = self._pixel_joints(joints)
        edges = self.edges
        a = joints[:, :, edges[:, 0], :]  # [B,T,E,2]
        b = joints[:, :, edges[:, 1], :]
        za = joint_depth[:, :, edges[:, 0]]
        zb = joint_depth[:, :, edges[:, 1]]

        h, w = self.image_size
        yy, xx = torch.meshgrid(
            torch.arange(h, device=joints.device, dtype=joints.dtype),
            torch.arange(w, device=joints.device, dtype=joints.dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1)[None, None, None]  # [1,1,1,H,W,2]
        av = a[..., None, None, :]
        ab = (b - a)[..., None, None, :]
        denom = ab.square().sum(dim=-1).clamp_min(self.eps)
        along = ((grid - av) * ab).sum(dim=-1) / denom
        along = along.clamp(0.0, 1.0)
        closest = av + along[..., None] * ab
        distance = (grid - closest).square().sum(dim=-1).clamp_min(self.eps).sqrt()

        radius = self.bone_radii.to(dtype=joints.dtype)[None, None, :, None, None]
        coverage = torch.sigmoid((radius - distance) / self.edge_softness)

        # Interpolate each capsule's depth, then use a coverage-aware soft
        # z-buffer. Background is handled by alpha rather than competing in
        # the depth softmax.
        bone_depth = za[..., None, None] + along * (zb - za)[..., None, None]
        logits = torch.log(coverage.clamp_min(self.eps)) - bone_depth / self.depth_temperature
        ownership = torch.softmax(logits, dim=2)
        alpha = 1.0 - torch.prod(1.0 - coverage.clamp(max=1.0 - self.eps), dim=2, keepdim=True)
        parts = ownership * alpha

        colors = self.bone_colors.to(dtype=joints.dtype)[None, None, :, :, None, None]
        rgb = (parts[:, :, :, None] * colors).sum(dim=2)
        rgba = torch.cat((rgb, alpha), dim=2)
        return {
            "rgba": rgba,
            "rgb": rgb,
            "alpha": alpha,
            "parts": parts,
            "coverage": coverage,
            "bone_depth": bone_depth,
        }
