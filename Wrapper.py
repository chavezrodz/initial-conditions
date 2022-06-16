import numpy as np
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
                criterion, 
                lr, 
                amsgrad,
                ):

        super().__init__()
        self.core_model = core_model
        self.norms = norms
        self.x_only = x_only
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
        abc = torch.permute(x, (0, -2,-1,-3))
        abc = (abc - self.norms[0].type_as(x))/self.norms[1].type_as(x)
        abc = (torch.permute(abc, (0, -1, -3, -2))).float()
        return abc

    def unscale(self, x):
        pass

    def forward(self, x):
        x = torch.cat(x, dim=-3)
        return self.core_model(x)

    def predict_step(self, batch, batch_idx):
        abc = self.scale(batch)
        a, b, c = abc[:, :16], abc[:, 16:32], abc[:, 32:48]
        x, y = (a, b), c
        if self.x_only:
            x, y = (a[..., 8:, :, :], b[..., 8:, :, :]), c[..., 8:, :, :]
        pred = self.forward(x)

        return pred, y

    def training_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        batch_size = y.shape[0]
        metrics = self.get_metrics(pred.flatten(-3, -1), y.flatten(-3, -1))
        self.log_dict(
            {f'{k}/train': v for k, v in metrics.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )
        return metrics[self.criterion]

    def validation_step(self, batch, batch_idx):
        pred, y = self.predict_step(batch, batch_idx)
        batch_size = y.shape[0]
        metrics = self.get_metrics(pred.flatten(-3, -1), y.flatten(-3, -1))
        self.log_dict(
            {f'{k}/validation': v for k, v in metrics.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )
        return metrics[self.criterion]
