import os
import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchvision.utils import make_grid

from src.models.flow import init_flow
from src.models.dist import recon_loss_fn
from src.models.encoder import init_encoder
from src.models.decoder import init_decoder


class NFVAE_GLMM(pl.LightningModule):
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
        self.encoder_logvar = nn.Linear(hdim, D)

        self.decoder = init_decoder(**decoder_configs)
        self.recon_loss = recon_loss_fn(**dist_configs)()
        self.flow = init_flow(**flow_configs)

    def forward(self, x):
        out = self.encoder(x)
        mu, log_var = self.encoder_mu(out), self.encoder_logvar(out)
        log_var = torch.clip(log_var, -10, 10)
        kl = -0.5 * (1. + log_var - mu ** 2. - torch.exp(log_var)).sum(-1)

        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

        logdet = 0
        for i, f in enumerate(self.flow):
            z, logdet_i = f(z)

            if i == 0:
                logdet = logdet_i
            else:
                logdet += logdet_i

        return z, kl, logdet

    def shared_step(self, x):
        zK, kl, logdet = self(x)

        xhat = self.decoder(zK)
        recon_loss = self.recon_loss(x, xhat)

        return xhat, (kl, recon_loss, - logdet)

    def training_step(self, batch, batch_idx):
        X, y = batch
        _, (kl, recon_loss, neglogdet) = self.shared_step(X)

        loss = torch.mean(kl + recon_loss + neglogdet)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_kl", torch.mean(kl), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_recon_loss", torch.mean(recon_loss), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if isinstance(neglogdet, torch.Tensor):
            self.log(
                "train_neglogdet", torch.mean(neglogdet), on_step=True, on_epoch=False, prog_bar=True, logger=True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        if self.hparams.simulation and self.current_epoch % 10 == 0:
            "For 1d/2d simulation cases, we save the sampled result"
            z = torch.distributions.Normal(
                loc=0., scale=1.
            ).rsample((1_000, self.hparams.D)).to(self.device)

            for i, f in enumerate(self.flow):
                os.makedirs(os.path.join(self.trainer.log_dir, 'uhat'), exist_ok=True)
                np.save(os.path.join(self.trainer.log_dir, f'uhat/Uhat-{self.current_epoch}-{i}'), z.cpu().numpy())
                z, _ = f(z)

            u = self.decoder(z)
            os.makedirs(os.path.join(self.trainer.log_dir, 'uhat'), exist_ok=True)
            np.save(os.path.join(self.trainer.log_dir, f'uhat/Uhat-{self.current_epoch}'), u.cpu().numpy())

            os.makedirs(os.path.join(self.trainer.log_dir, 'beta'), exist_ok=True)
            beta = self.recon_loss.fe.weight.detach()
            np.save(os.path.join(self.trainer.log_dir, f'beta/beta-{self.current_epoch}'), beta.cpu().numpy())

        elif not self.hparams.simulation:
            Xhat, loss = self.shared_step(X)
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            if batch_idx == 0 and self.current_epoch % 20 == 0:
                Xhat = Xhat.reshape([-1] + list(self.hparams.data_shape))
                self.logger.log_image(
                    key="Recon_image", images=[make_grid(Xhat[:36], nrow=6, normalize=True)]
                )

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError
        return optimizer
