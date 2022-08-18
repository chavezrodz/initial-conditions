import numpy as np
from torch import norm
import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule


class Wrapper(LightningModule):  
    def __init__(self,
                core_model,
                norms,
                criterion,
                lr, 
                amsgrad,
                ):

        super().__init__()
        self.core_model = core_model
        self.norms = norms
        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad
            )
        return optimizer

    def get_metrics(self, pred, y):
        """
        compute element metric first, final mean taken in dict
        """

        # per plane
        sq_err = torch.square(pred - y).sum(dim=(2,3))/torch.square(y).sum(dim=(2,3))
        # per feature
        sq_err = sq_err.mean(dim=1)

        abs_err = (pred - y).abs().sum(dim=(2,3))/y.abs().sum(dim=(2,3))
        abs_err = abs_err.mean(dim=1)

        sum_err = (pred.sum(dim=(2, 3)) - y.sum(dim=(2, 3)))/y.sum(dim=(2, 3))
        sum_err = sum_err.abs().mean(dim=1)

        # Per batch averages
        metrics = dict(
            sq_err=sq_err.mean(),
            sum_err=sum_err.mean(),
            abs_err=abs_err.mean()

        )
        return metrics

    def scale(self, x):
        norms = self.norms.type_as(x)
        abc = torch.permute(x, (0, -2,-1,-3))
        abc = (abc - norms[0])/norms[1]
        abc = (torch.permute(abc, (0, -1, -3, -2))).float()
        return abc

    def forward(self, x):
        x = torch.cat(x, dim=-3)
        return self.core_model(x)

    def predict_step(self, batch, batch_idx):
        # scale everything, including targets for loss calc
        data, file_nb = batch
        abc = self.scale(data)
        # combining inputs, doubling inputs for symetry ab, ba
        a, b, c = abc[:, :16], abc[:, 16:32], abc[:, 32:48]

        # double data by symetry to enforce abellian func:
        a, b = torch.cat([a, b], axis=0), torch.cat([b, a], axis=0)
        c = torch.cat([c, c], axis=0)

        # Commented out because no need to check doubled data on test
        # file_nb = (*file_nb, *file_nb)

        x, y = (a, b), c
        pred = self.forward(x)

        return pred, y, file_nb

    def training_step(self, batch, batch_idx):
        pred, y, _ = self.predict_step(batch, batch_idx)
        batch_size = y.shape[0]

        metrics_scaled = self.get_metrics(pred, y)
        self.log_dict(
            {f'train/{k}': v for k, v in metrics_scaled.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )

        return metrics_scaled[self.criterion]

    def validation_step(self, batch, batch_idx):
        pred, y, _ = self.predict_step(batch, batch_idx)
        batch_size = y.shape[0]
        metrics_scaled = self.get_metrics(pred, y)
        self.log_dict(
            {f'validation/{k}': v for k, v in metrics_scaled.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )

        return metrics_scaled[self.criterion]

    def test_step(self, batch, batch_idx):
        pred, y, _ = self.predict_step(batch, batch_idx)
        batch_size = y.shape[0]

        metrics_scaled = self.get_metrics(pred, y)
        self.log_dict(
            {f'test/{k}': v for k, v in metrics_scaled.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )
        return metrics_scaled[self.criterion]
