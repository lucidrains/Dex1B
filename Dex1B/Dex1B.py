import torch
from torch.nn import Module, ModuleList

from x_transformers import Encoder

from x_mlps_pytorch import MLP

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class PointTransformer(Module):
    """ https://arxiv.org/abs/2312.10035v1 """

    def __init__(self):
        super().__init__()

class CVAE(Module):
    def __init__(
        self,
        dim,
        dim_hiddens = (256, 512, 256), # from Table 6. in paper
        kl_loss_weight = 1e-4
    ):
        super().__init__()
        assert len(dim_hiddens) > 0
        dim_latent = default_layer_sizes[-1]

        self.encode = MLP(dim, *default_layer_sizes)

        self.to_mean_log_variance = nn.Linear(dim_latent, dim_latent * 2, bias = False)

        self.decode = MLP(*default_layer_sizes, dim)

        # loss weights

        self.kl_loss_weight = kl_loss_weight

    def forward(
        self,
        inp, # (b d)
        return_loss = False
    ):

        encoded = self.encode(inp)

        mean, log_variance = self.to_mean_log_variance(encoded).chunk(2, dim = -1)

        std = (0.5 * log_variance).exp()

        noise = torch.randn_like(mean)

        reparamed = mean + std * noise

        recon = self.decode(reparamed)

        if not return_loss:
            return recon

        mse_loss = F.mse_loss(recon, inp)

        kl_loss = 0.5 * (mean.square() + log_variance.exp() - log_variance - 1.).sum(dim = -1).mean()

        total_loss = (
            mse_loss +
            kl_loss * self.kl_loss_weight
        )

        loss_breakdown = (mse_loss, kl_loss)

        return total_loss, loss_breakdown

class DexSimple(Module):
    def __init__(
        self,
        point_transformer: PointTransformer,
        cvae: CVAE
    ):
        super().__init__()

        self.pointnet = pointnet
        self.cvae = cvae
