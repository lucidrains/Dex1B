import pytest

import torch

def test_sdf():
    from Dex1B.Dex1B import simple_sdf_loss

    surface_points = torch.randn(2, 1024, 3).requires_grad_()
    mask = torch.randint(0, 2, (2, 1024)).bool()

    hand_points = torch.randn(2, 16, 3)
    hand_point_radius = torch.rand(2, 16)

    sdf_loss = simple_sdf_loss(surface_points, hand_points, hand_point_radius, mask)
    sdf_loss.backward()

    assert sdf_loss.numel() == 1
