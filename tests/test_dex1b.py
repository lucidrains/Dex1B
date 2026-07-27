import pytest

import torch
from torch import nn
from torch.nn import Module

from Dex1B import (
    HandGeometry,
    PointNet,
    PointTransformer,
    DexSimple,
    simple_sdf_loss,
    contact_distance_loss,
    smoothness_loss,
    joint_limit_loss,
    self_penetration_loss,
    associate_pose_with_object_point,
    debias_sample,
    post_optimize
)

class MockHandFK(Module):
    def __init__(self, dim_pose, num_spheres = 16, num_contacts = 5):
        super().__init__()
        self.to_spheres = nn.Linear(dim_pose, num_spheres * 3)
        self.to_contacts = nn.Linear(dim_pose, num_contacts * 3)
        self.register_buffer('radii', torch.rand(num_spheres) * 1e-2)
        self.register_buffer('link_ids', torch.randint(0, 4, (num_spheres,)))

    def forward(self, pose):
        batch = pose.shape[0]
        centers = self.to_spheres(pose).reshape(batch, -1, 3)
        contacts = self.to_contacts(pose).reshape(batch, -1, 3)
        radii = self.radii.expand(batch, -1)
        link_ids = self.link_ids.expand(batch, -1)
        return HandGeometry(centers, radii, contacts, link_ids)

def test_encoders():
    points = torch.randn(2, 64, 3)
    mask = torch.randint(0, 2, (2, 64)).bool()

    pointnet = PointNet(dim = 128)
    global_feat, local_feat = pointnet(points, mask = mask)
    assert global_feat.shape == (2, 128)
    assert local_feat.shape == (2, 64, 128)

    point_transformer = PointTransformer(dim = 128)
    global_feat_t, local_feat_t = point_transformer(points, mask = mask)
    assert global_feat_t.shape == (2, 128)
    assert local_feat_t.shape == (2, 64, 128)

def test_losses():
    surface_points = torch.randn(2, 1024, 3).requires_grad_()
    mask = torch.randint(0, 2, (2, 1024)).bool()

    hand_points = torch.randn(2, 16, 3)
    hand_point_radius = torch.rand(2, 16)

    sdf_loss = simple_sdf_loss(surface_points, hand_points, hand_point_radius, mask)
    sdf_loss.backward()
    assert sdf_loss.numel() == 1

    contact_points = torch.randn(2, 5, 3)
    c_loss = contact_distance_loss(surface_points, contact_points, mask = mask)
    assert c_loss.numel() == 1

    poses = torch.randn(2, 4, 25)
    s_loss = smoothness_loss(poses)
    assert s_loss.numel() == 1

    joints = torch.randn(2, 25)
    j_limits = (torch.full((25,), -1.), torch.full((25,), 1.))
    j_loss = joint_limit_loss(joints, j_limits)
    assert j_loss.numel() == 1

    sphere_centers = torch.randn(2, 16, 3)
    sphere_radii = torch.rand(16) * 0.05
    sphere_link_ids = torch.randint(0, 4, (16,))
    self_loss = self_penetration_loss(sphere_centers, sphere_radii, sphere_link_ids)
    assert self_loss.numel() == 1

@pytest.mark.parametrize('num_frames', (1, 4))
@pytest.mark.parametrize('condition_on_local_point_feature', (True, False))
@pytest.mark.parametrize('encoder_type', ('transformer', 'pointnet'))
def test_dex_simple(num_frames, condition_on_local_point_feature, encoder_type):
    dim_pose = 25 # 3 + 3 + 19 dof

    if encoder_type == 'pointnet':
        point_encoder = PointNet(dim = 256)
    else:
        point_encoder = PointTransformer(dim = 256)

    model = DexSimple(
        dim_pose = dim_pose,
        num_frames = num_frames,
        point_encoder = point_encoder,
        condition_on_local_point_feature = condition_on_local_point_feature,
        hand_fk = MockHandFK(dim_pose)
    )

    hand_pose = torch.randn(2, num_frames, dim_pose)
    points = torch.randn(2, 128, 3)
    mask = torch.randint(0, 2, (2, 128)).bool()
    assoc_point_indices = torch.randint(0, 128, (2,)) if condition_on_local_point_feature else None

    loss, (mse, kl, sdf, dist, smooth) = model(
        hand_pose,
        points,
        mask = mask,
        assoc_point_indices = assoc_point_indices,
        return_loss_breakdown = True
    )

    loss.backward()

    sampled = model.sample(points, mask = mask, assoc_point_indices = assoc_point_indices)
    assert sampled.shape == (2, num_frames, dim_pose)

def test_associate_and_debias():
    points = torch.randn(2, 128, 3)
    mask = torch.randint(0, 2, (2, 128)).bool()

    palm_center = torch.randn(2, 3)
    thumb_tip = torch.randn(2, 3)
    middle_finger_tip = torch.randn(2, 3)

    indices = associate_pose_with_object_point(palm_center, thumb_tip, middle_finger_tip, points, mask = mask)
    assert indices.shape == (2,) and (indices < 128).all()

    counts = torch.randint(0, 10, (128,))
    sampled_indices = debias_sample(counts, 32)
    assert sampled_indices.shape == (32,) and (sampled_indices < 128).all()

def test_post_optimize():
    dim_pose = 25
    hand_fk = MockHandFK(dim_pose)

    hand_pose = torch.randn(2, dim_pose)
    surface_points = torch.randn(2, 128, 3)

    joint_limits = (torch.full((dim_pose,), -1.), torch.full((dim_pose,), 1.))

    refined = post_optimize(
        hand_pose,
        hand_fk,
        surface_points,
        joint_limits = joint_limits,
        steps = 2
    )

    assert refined.shape == hand_pose.shape and not refined.requires_grad
