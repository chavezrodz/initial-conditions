import os
from argparse import ArgumentParser
from dataloaders import get_iterators
from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from Model import Model
from UNET import UNET
from Wrapper import Wrapper
from utils import make_file_prefix 
from memory_profiler import profile

@profile
def main(args):
    utilities.seed.seed_everything(seed=args.seed, workers=True)

    devices = 'auto'
    n_workers = args.num_workers

    if args.logger == 'tb':
        logger = TensorBoardLogger(
            save_dir=os.path.join(args.results_dir, "TB_logs"),
            name=make_file_prefix(args),
            default_hp_metric=True
            )
    elif args.logger == 'wandb':
        wandb.init()
        wandb.config.update(args)
        logger = WandbLogger(
            project="IC",
            save_dir=os.path.join(args.results_dir, "wandb")
            )
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
        logger=logger,
        accelerator='auto',
        devices=devices,
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        )

    (train, val, test_dl), norms = get_iterators(
        datapath=args.datapath,
        cached=args.cached,
        batch_size=args.batch_size,
        n_workers=n_workers
        )

    # Only x channels
    # 8 per dim, two dims, two inputs (a&b)
    input_dim, output_dim = 32, 16

    if args.model == 'MLP': 
        model = Model(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            output_dim=output_dim,
            )
    elif args.model == 'UNET':
        model = UNET(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            output_dim=output_dim,
            )


    Wrapped_Model = Wrapper(
        model,
        norms,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad
        )


    trainer.fit(
        Wrapped_Model,
        train,
        val,
        ckpt_path='.'
        )

    trainer.test(Wrapped_Model, test_dl)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--model", default='UNET', type=str,
                        choices=['MLP', 'UNET'])

    # data params
    parser.add_argument("--res", default='128x128', type=str)
    parser.add_argument("--energy", default='5020', type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--cached", default=True, type=bool)

    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='sq_err', type=str,
                        choices=['sum_err', 'abs_err', 'sq_err'])
    parser.add_argument("--add_sum_err", default=True, type=bool)

    parser.add_argument("--logger", default='tb', type=str, choices=['wandb', 'tb'])
    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args)
