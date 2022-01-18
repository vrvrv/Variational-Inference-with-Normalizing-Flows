from functools import partial
from typing import List, Optional
import os
import copy
import wandb
import math
import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchvision.utils import make_grid

from torchmetrics import FID, IS

from src.models.flow import init_flow
from src.models.dist import recon_loss_fn
from src.models.encoder import init_encoder
from src.models.decoder import init_decoder


class NFVAE(pl.LightningModule):
    def __init__(
            self,
            data_shape: list,
            D: int,
            encoder_configs: dict,
            decoder_configs: dict,
            dist_configs: dict,
            flow_configs: dict,
            **kwargs
    ):
        """
        :param T: total diffusion steps
        :param ch: base channel of UNet
        :param ch_mult: channel multiplier
        :param attn: add attention to these levels
        :param num_res_blocks: resblock in each level
        :param dropout: dropout rate of resblock
        :param beta_1: start beta value
        :param beta_T: end beta value
        :param var_type: variance type ('fixedlarge', 'fixedsmall')
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder, hdim = init_encoder(**encoder_configs)
        self.encoder_mu = nn.Linear(hdim, D)
        self.encoder_logstd = nn.Linear(hdim, D)

        self.decoder = init_decoder(**decoder_configs)
        self.recon_loss = recon_loss_fn(**dist_configs)
        self.flow = init_flow(**flow_configs)

    def forward(self, x):
        out = self.encoder(x)
        mu, log_std = self.encoder_mu(out), self.encoder_logstd(out)

        std = torch.exp(log_std)

        kl = 0.5 * (mu ** 2. + std ** 2. - 2 * log_std - 1).sum(-1)

        z = mu + std * torch.randn_like(mu)

        # log_pz0 = torch.sum(-0.5 * torch.log(torch.tensor(2 * math.pi)) - log_std - 0.5 * ((z - mu) / std) ** 2, axis=1)

        for i, f in enumerate(self.flow):
            z, logdet_i = f(z)

            if i == 0:
                logdet = logdet_i
            else:
                logdet += logdet_i

        # log_pzk = torch.sum(
        #     -0.5 * (torch.log(torch.tensor(2 * math.pi)) + z ** 2),
        #     axis=1
        # )
        #
        # kl = log_pz0 - log_pzk

        return z, kl, logdet

    def shared_step(self, x):
        zK, kl, logdet = self(x)

        xhat = self.decoder(zK)
        recon_loss = self.recon_loss(x, xhat)

        loss = torch.mean(kl + recon_loss - logdet)

        return xhat, loss

    def training_step(self, batch, batch_idx):
        X, y = batch
        _, loss = self.shared_step(X)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        Xhat, loss = self.shared_step(X)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 20 == 0:
            Xhat = Xhat.reshape([-1] + list(self.hparams.data_shape))
            self.logger.log_image(
                key="Recon_image", images=[make_grid(Xhat[:36], nrow=6, normalize=True)]
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer
