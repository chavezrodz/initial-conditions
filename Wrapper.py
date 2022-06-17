import numpy as np
from torch import norm
import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule

class MAPE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y):
        return ((y - y_pred)/y).abs().mean()


class Wrapper(LightningModule):  
    def __init__(self,
                core_model,
                norms, 
                x_only,
                double_data_by_sym,
                criterion, 
                lr, 
                amsgrad,
                ):

        super().__init__()
        self.core_model = core_model
        self.norms = norms
        self.x_only = x_only
        self.double_data_by_sym = double_data_by_sym
        self.criterion = criterion
        self.lr = lr
        self.amsgrad = amsgrad

        self.pc_err = MAPE()
        self.abs_err = nn.L1Loss()
        self.mse = nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad
            )
        return optimizer

    def get_metrics(self, pred, y):
        metrics = dict(
            abs_err=self.abs_err(pred, y),
            pc_err=self.pc_err(pred, y),
            mse=self.mse(pred, y),
        )
        return metrics

    def scale(self, x):
        norms = self.norms.type_as(x)
        abc = torch.permute(x, (0, -2,-1,-3))
        abc = (abc - norms[0])/norms[1]
        abc = (torch.permute(abc, (0, -1, -3, -2))).float()
        return abc

    def unscale(self, y):
        """
        outputs only
        """
        dims_used = y.shape[-3]
        norms = self.norms.type_as(y)
        if dims_used == 8:
            norms = norms[:, -16:-8]
        elif dims_used == 16:
            norms = norms[:, -16:]

        y = torch.permute(y, (0, -2,-1,-3))
        y = y*norms[1] + norms[0]
        y = (torch.permute(y, (0, -1, -3, -2)))
        return y

    def forward(self, x):
        x = torch.cat(x, dim=-3)
        return self.core_model(x)

    def predict_step(self, batch, batch_idx):
        # scale everything, including targets for loss calc
        abc = self.scale(batch)
        # combining inputs, doubling inputs for symetry ab, ba
        a, b, c = abc[:, :16], abc[:, 16:32], abc[:, 32:48]
        if self.double_data_by_sym:
            a, b = torch.cat([a, b], axis=0), torch.cat([b, a], axis=0)
            c = torch.cat([c, c], axis=0)
        x, y = (a, b), c
        # Slicing x components only
        if self.x_only:
            x, y = (a[..., 8:, :, :], b[..., 8:, :, :]), c[..., 8:, :, :]
        pred = self.forward(x)
        return pred, y

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        batch_size = y.shape[0]

        metrics_scaled = self.get_metrics(pred.flatten(-3, -1), y.flatten(-3, -1))
        self.log_dict(
            {f'{k}/validation/scaled': v for k, v in metrics_scaled.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )

        return metrics_scaled[self.criterion]

    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        batch_size = y.shape[0]

        metrics_scaled = self.get_metrics(pred.flatten(-3, -1), y.flatten(-3, -1))
        self.log_dict(
            {f'{k}/validation/scaled': v for k, v in metrics_scaled.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )

        pred, y = self.unscale(pred), self.unscale(y)
        metrics_unscaled = self.get_metrics(pred.flatten(-3, -1), y.flatten(-3, -1))
        self.log_dict(
            {f'{k}/validation/unscaled': v for k, v in metrics_unscaled.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )

        return metrics_scaled[self.criterion]
