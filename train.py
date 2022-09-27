import os
import wandb
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from Datamodule import DataModule
from utils import make_file_prefix, load_model
from memory_profiler import profile


# @profile
def main(args):
    utilities.seed.seed_everything(seed=args.seed, workers=True)
    dm = DataModule(args, stage='train')
    dm.setup(stage="init")
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(args.results_dir, "tb_logs"),
        name=make_file_prefix(args),
        default_hp_metric=True
        )
    loggers = [tb_logger]#, wb_logger]
    for logger in loggers:
        logger.log_hyperparams(
                args
                )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.results_dir, 'saved_models'),
        save_top_k=1,
        monitor='validation/'+args.criterion,
        mode="min",
        filename=make_file_prefix(args)+'_val_err_{validation/sq_err:.2e}',
        auto_insert_metric_name=False,
        save_last=False
        )

    trainer = Trainer(
        logger=loggers,
        accelerator='auto',
        devices='auto',
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        auto_lr_find=True
        )

    model = load_model(args, dm)

    trainer.fit(
        model,
        datamodule=dm,
        )

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--model", default='UNET', type=str,
                        choices=['MLP', 'UNET'])

    # data params
    parser.add_argument("--max_samples", default=-1, type=int)
    parser.add_argument("--train_res", default='512x512', type=str, choices=['128x128', '512x512'])
    parser.add_argument("--train_energy", default='all', type=str,
                        choices=['193', '2760', '5020', 'all'])

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--cached", default=False, type=bool)

    parser.add_argument("--data_pc_norms", default=1, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='sq_err', type=str,
                        choices=['sum_err', 'abs_err', 'sq_err'])
    parser.add_argument("--add_sum_err", default=True, type=bool)

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='fakedata', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args)
