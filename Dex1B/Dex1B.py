from __future__ import annotations
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import cdist, cat, arange, randn, randn_like
from torch.nn import Module, Linear
from torch.optim import Adam

from torch_einops_utils import safe_cat, detach_tensor

from x_transformers import Encoder
from x_mlps_pytorch import MLP

import einx
from einops import rearrange, repeat, einsum

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# hand geometry, returned by a hand-specific differentiable forward kinematics
# spheres approximate the hand links (~10 per link), contact points are predefined points on the fingers

HandGeometry = namedtuple('HandGeometry', [
    'sphere_centers',   # (b m 3)
    'sphere_radii',     # (b m) | (m)
    'contact_points',   # (b k 3) | None
    'sphere_link_ids'   # (b m) | (m) | None
], defaults = (None, None))

# losses

def simple_sdf_loss(
    surface_points,     # (b n 3)
    hand_points,        # (b m 3)
    hand_point_radius,  # (b m) | (m)
    mask = None         # (b n) | None
):
    """
    their simple point-sphere penetration loss in section IV
    L_sdf = Σ_c max(0, r_c - min_p ||c - p||)²
    they claim this to be more stable than mesh-sphere
    """

    dist = cdist(hand_points, surface_points)

    if exists(mask):
        mask_value = torch.finfo(dist.dtype).max
        dist = einx.where('b n, b m n,', mask, dist, mask_value)

    closest_dist = dist.amin(dim = -1)

    penetration = (hand_point_radius - closest_dist).relu()

    return penetration.square().sum(dim = -1).mean()

def contact_distance_loss(
    surface_points,     # (b n 3)
    contact_points,     # (b k 3)
    mask = None         # (b n) | None
):
    """
    L_D - encourages predefined hand contact points to touch the object surface
    """

    dist = cdist(contact_points, surface_points)

    if exists(mask):
        mask_value = torch.finfo(dist.dtype).max
        dist = einx.where('b n, b k n,', mask, dist, mask_value)

    return dist.amin(dim = -1).sum(dim = -1).mean()

def smoothness_loss(
    poses               # (b nf d)
):
    """
    E_smooth = Σ_t ||g_t - g_t-1||²
    """

    return poses.diff(dim = 1).square().sum(dim = (-1, -2)).mean()

def joint_limit_loss(
    joints,             # (b d)
    joint_limits        # ((d), (d)) - (lower, upper)
):
    lower, upper = joint_limits
    return ((joints - upper).relu() + (lower - joints).relu()).sum(dim = -1).mean()

def self_penetration_loss(
    sphere_centers,         # (b m 3)
    sphere_radii,           # (b m) | (m)
    sphere_link_ids = None  # (b m) | (m) | None
):
    """
    E_s - avoids self-penetration between hand spheres on different links
    """

    dist = cdist(sphere_centers, sphere_centers)

    overlap = (einx.add('... i, ... j -> ... i j', sphere_radii, sphere_radii) - dist).relu()

    eye = torch.eye(dist.shape[-1], device = dist.device, dtype = torch.bool)
    exclude = eye

    if exists(sphere_link_ids):
        same_link = einx.equal('... i, ... j -> ... i j', sphere_link_ids, sphere_link_ids)
        exclude = exclude | same_link

    overlap = overlap.masked_fill(exclude, 0.)

    return overlap.sum(dim = (-1, -2)).mean()

# associating a hand pose with a single object point (section III - debias conditioning)
# heading direction v goes from palm center to midpoint between thumb tip and middle finger tip
# the closest object point along this ray is the associated point

def associate_pose_with_object_point(
    palm_center,        # (b 3)
    thumb_tip,          # (b 3)
    middle_finger_tip,  # (b 3)
    points,             # (b n 3)
    mask = None         # (b n) | None
):
    heading = l2norm((thumb_tip + middle_finger_tip) / 2. - palm_center)

    rel = einx.subtract('b n c, b c', points, palm_center)

    proj = einsum(rel, heading, 'b n c, b c -> b n')

    perp = rel - einx.multiply('b n, b c -> b n c', proj, heading)
    perp_dist = perp.norm(dim = -1)

    invalid = proj <= 0.

    if exists(mask):
        invalid = invalid | ~mask

    mask_value = torch.finfo(perp_dist.dtype).max
    scores = perp_dist.masked_fill(invalid, mask_value)

    # fallback to plain closest point if nothing lies along the heading direction

    dist = rel.norm(dim = -1)

    if exists(mask):
        dist = dist.masked_fill(~mask, mask_value)

    all_invalid = invalid.all(dim = -1)
    scores = einx.where('b, b n, b n', all_invalid, dist, scores)

    return scores.argmin(dim = -1)

# debiased sampling - sample associated points inversely proportional to how often
# they already appear in the dataset, expanding diversity beyond the seed distribution

def debias_sample(
    counts,             # (n) counts of existing actions per object point (or per object)
    num_samples
):
    weights = (counts.float() + 1.).reciprocal()
    return torch.multinomial(weights, num_samples, replacement = True)

# visual encoders

class PointNet(Module):
    """
    visual encoder using simple pointnet MLPs
    returns the global object feature f_obj and local per-point features f_p
    """

    def __init__(
        self,
        dim = 256,
        dim_input = 3,
        embed_dim_hiddens = (64, 128)
    ):
        super().__init__()
        self.mlp = MLP(dim_input, *embed_dim_hiddens, dim)

    def forward(
        self,
        points,         # (b n 3)
        mask = None     # (b n) | None
    ):
        local_features = self.mlp(points)

        pooled = local_features

        if exists(mask):
            pooled = einx.where('b n, b n d,', mask, pooled, max_neg_value(pooled))

        global_feature = pooled.amax(dim = 1)

        return global_feature, local_features

class PointTransformer(Module):
    """
    stand-in for PointNet in the paper, using a simple point transformer
    https://arxiv.org/abs/2312.10035
    returns the global object feature f_obj and local per-point features f_p
    """

    def __init__(
        self,
        dim = 256,
        dim_input = 3,
        depth = 2,
        heads = 8,
        dim_head = 64,
        embed_dim_hiddens = (64, 128),
        **encoder_kwargs
    ):
        super().__init__()
        self.to_tokens = MLP(dim_input, *embed_dim_hiddens, dim)

        self.encoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **encoder_kwargs
        )

    def forward(
        self,
        points,         # (b n 3)
        mask = None     # (b n) | None
    ):
        tokens = self.to_tokens(points)

        local_features = self.encoder(tokens, mask = mask)

        pooled = local_features

        if exists(mask):
            pooled = einx.where('b n, b n d,', mask, pooled, max_neg_value(pooled))

        global_feature = pooled.amax(dim = 1)

        return global_feature, local_features

# cvae

class CVAE(Module):
    def __init__(
        self,
        dim,
        dim_cond = 0,
        dim_hiddens = (256, 512, 256), # from Table 6. in paper
        kl_loss_weight = 1e-4
    ):
        super().__init__()
        assert len(dim_hiddens) > 0
        dim_latent = dim_hiddens[-1]

        self.dim_cond = dim_cond
        self.dim_latent = dim_latent

        self.encode = MLP(dim + dim_cond, *dim_hiddens)

        self.to_mean_log_variance = Linear(dim_latent, dim_latent * 2, bias = False)

        self.decode = MLP(dim_latent + dim_cond, *dim_hiddens, dim)

        # loss weights

        self.kl_loss_weight = kl_loss_weight

    @property
    def device(self):
        return self.to_mean_log_variance.weight.device

    def sample(
        self,
        cond = None,        # (b dim_cond) | None
        batch_size = 1,
        temperature = 1.
    ):
        if exists(cond):
            batch_size = cond.shape[0]

        latents = randn(batch_size, self.dim_latent, device = self.device) * temperature

        return self.decode(safe_cat((latents, cond), dim = -1))

    def forward(
        self,
        inp,                # (b d)
        cond = None,        # (b dim_cond) | None
        return_loss = False
    ):
        encoded = self.encode(safe_cat((inp, cond), dim = -1))

        mean, log_variance = self.to_mean_log_variance(encoded).chunk(2, dim = -1)

        std = (0.5 * log_variance).exp()

        reparamed = mean + std * randn_like(mean)

        recon = self.decode(safe_cat((reparamed, cond), dim = -1))

        if not return_loss:
            return recon

        mse_loss = F.mse_loss(recon, inp)

        kl_loss = 0.5 * (mean.square() + log_variance.exp() - log_variance - 1.).sum(dim = -1).mean()

        total_loss = (
            mse_loss +
            kl_loss * self.kl_loss_weight
        )

        loss_breakdown = (mse_loss, kl_loss)

        return total_loss, loss_breakdown, recon

class DexSimple(Module):
    """
    section IV - a CVAE over hand poses g = (T, R, θ) flattened to (nf × dof),
    conditioned on the object point cloud feature (and optionally the local
    feature of the associated object point), trained with mse + kl + sdf +
    contact distance + smoothness losses
    """

    def __init__(
        self,
        dim_pose,                                   # dof of the hand pose per frame
        num_frames = 1,
        dim = 256,
        point_encoder: Module | dict = dict(),
        point_transformer: Module | dict | None = None, # for backward compatibility
        cvae_dim_hiddens = (256, 512, 256),
        condition_on_local_point_feature = True,
        hand_fk: Module | None = None,              # (b dof) -> HandGeometry, differentiable forward kinematics
        kl_loss_weight = 1e-4,                      # loss weights from Table 6.
        sdf_loss_weight = 1e-4,
        distance_loss_weight = 1e-4,
        smoothness_loss_weight = 1e-5
    ):
        super().__init__()
        self.dim_pose = dim_pose
        self.num_frames = num_frames

        if exists(point_transformer):
            point_encoder = point_transformer

        if isinstance(point_encoder, dict):
            point_encoder = PointTransformer(dim = dim, **point_encoder)

        self.point_encoder = point_encoder

        self.condition_on_local_point_feature = condition_on_local_point_feature

        dim_cond = dim * (2 if condition_on_local_point_feature else 1)

        self.cvae = CVAE(
            dim_pose * num_frames,
            dim_cond = dim_cond,
            dim_hiddens = cvae_dim_hiddens,
            kl_loss_weight = kl_loss_weight
        )

        self.hand_fk = hand_fk

        # loss weights

        self.sdf_loss_weight = sdf_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.smoothness_loss_weight = smoothness_loss_weight

    @property
    def point_transformer(self):
        return self.point_encoder

    def get_condition(
        self,
        points,                     # (b n 3)
        mask = None,                # (b n) | None
        assoc_point_indices = None  # (b) | None
    ):
        global_feature, local_features = self.point_encoder(points, mask = mask)

        if not self.condition_on_local_point_feature:
            return global_feature

        assert exists(assoc_point_indices), '`assoc_point_indices` must be passed in when conditioning on the local associated point feature'

        batch = points.shape[0]
        point_feature = local_features[arange(batch, device = points.device), assoc_point_indices]

        return cat((global_feature, point_feature), dim = -1)

    @torch.no_grad()
    def sample(
        self,
        points,                     # (b n 3)
        mask = None,                # (b n) | None
        assoc_point_indices = None, # (b) | None
        temperature = 1.
    ):
        cond = self.get_condition(points, mask = mask, assoc_point_indices = assoc_point_indices)

        pose = self.cvae.sample(cond = cond, temperature = temperature)

        return rearrange(pose, 'b (nf d) -> b nf d', nf = self.num_frames)

    def forward(
        self,
        hand_pose,                  # (b nf d) | (b d)
        points,                     # (b n 3)
        mask = None,                # (b n) | None
        assoc_point_indices = None, # (b) | None
        return_loss_breakdown = False
    ):
        if hand_pose.ndim == 2:
            hand_pose = rearrange(hand_pose, 'b d -> b 1 d')

        batch, num_frames = hand_pose.shape[:2]
        assert num_frames == self.num_frames

        flattened_pose = rearrange(hand_pose, 'b nf d -> b (nf d)')

        cond = self.get_condition(points, mask = mask, assoc_point_indices = assoc_point_indices)

        cvae_loss, (mse_loss, kl_loss), recon = self.cvae(flattened_pose, cond = cond, return_loss = True)

        recon = rearrange(recon, 'b (nf d) -> b nf d', nf = num_frames)

        zero = flattened_pose.new_tensor(0.)
        sdf_loss = distance_loss = smooth_loss = zero

        # geometric losses on the reconstruction, through differentiable forward kinematics

        if exists(self.hand_fk):
            frames_recon = rearrange(recon, 'b nf d -> (b nf) d')

            frames_points = repeat(points, 'b ... -> (b nf) ...', nf = num_frames)
            frames_mask = repeat(mask, 'b ... -> (b nf) ...', nf = num_frames) if exists(mask) else None

            geometry = self.hand_fk(frames_recon)

            sdf_loss = simple_sdf_loss(frames_points, geometry.sphere_centers, geometry.sphere_radii, mask = frames_mask)

            if exists(geometry.contact_points):
                distance_loss = contact_distance_loss(frames_points, geometry.contact_points, mask = frames_mask)

        if num_frames > 1:
            smooth_loss = smoothness_loss(recon)

        total_loss = (
            cvae_loss +
            sdf_loss * self.sdf_loss_weight +
            distance_loss * self.distance_loss_weight +
            smooth_loss * self.smoothness_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (mse_loss, kl_loss, sdf_loss, distance_loss, smooth_loss)

        return total_loss, loss_breakdown

# post optimization (section III)
# E_post = w_dis * E_dis + w_sdf * E_sdf + w_j * E_j + w_s * E_s
# ~100 steps of slight finger adjustments on sampled poses before simulator validation

def post_optimize(
    hand_pose,              # (b d)
    hand_fk,                # (b d) -> HandGeometry
    surface_points,         # (b n 3)
    mask = None,            # (b n) | None
    joint_limits = None,    # ((d), (d)) | None
    pose_to_joints = None,  # (b d) -> (b dj) | None - defaults to identity
    steps = 100,
    lr = 1e-3,
    distance_loss_weight = 100.,
    sdf_loss_weight = 100.,
    self_penetration_loss_weight = 10.,
    joint_limit_loss_weight = 1.
):
    pose_to_joints = default(pose_to_joints, lambda pose: pose)

    pose = detach_tensor(hand_pose, clone = True).requires_grad_()

    optimizer = Adam([pose], lr = lr)

    for _ in range(steps):
        geometry = hand_fk(pose)

        loss = sdf_loss_weight * simple_sdf_loss(surface_points, geometry.sphere_centers, geometry.sphere_radii, mask = mask)

        if exists(geometry.contact_points):
            loss = loss + distance_loss_weight * contact_distance_loss(surface_points, geometry.contact_points, mask = mask)

        loss = loss + self_penetration_loss_weight * self_penetration_loss(geometry.sphere_centers, geometry.sphere_radii, geometry.sphere_link_ids)

        if exists(joint_limits):
            loss = loss + joint_limit_loss_weight * joint_limit_loss(pose_to_joints(pose), joint_limits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return pose.detach()
