import torch.optim
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_forecasting.metrics import MAPE


class Wrapper(LightningModule):  
    def __init__(self,
                core_model,
                x_only,
                criterion, 
                lr, 
                amsgrad,
                ):

        super().__init__()
        self.core_model = core_model
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

    def forward(self, x):
        x = torch.cat(x, dim=-3)
        return self.core_model(x)

    def training_step(self, batch, batch_idx):
        a,b,c = batch
        x, y = (a, b), c
        if self.x_only:
            x, y = (a[..., 8:, :, :], b[..., 8:, :, :]), c[..., 8:, :, :]
        batch_size = y.shape[0]
        pred = self.forward(x)
        metrics = self.get_metrics(pred.flatten(-3, -1), y.flatten(-3, -1))
        self.log_dict(
            {f'{k}/train': v for k, v in metrics.items()},
            on_epoch=True, on_step=False, batch_size=batch_size
            )
        return metrics[self.criterion]

