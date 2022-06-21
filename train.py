import os
from argparse import ArgumentParser
from dataloaders import get_iterators
from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb
from Model import Model
from UNET import UNET
from Wrapper import Wrapper

def main(args):
    utilities.seed.seed_everything(seed=args.seed, workers=True)

    if args.gpu:
        avail_gpus = 1
        n_workers = 0
    else:
        avail_gpus = 0
        n_workers = 8

    # Only x channels
    input_dim = 8 if args.x_only else 16
    input_dim *= 2 # concatenating A&B
    output_dim = 8 if args.x_only else 16

    (train, val, test_dl), norms = get_iterators(
        datapath=args.datapath,
        cached=args.cached,
        batch_size=args.batch_size,
        n_workers=n_workers
        )

    if args.logger == 'tb':
        logger = TensorBoardLogger(
            save_dir=os.path.join(args.results_dir, "TB_logs"),
            default_hp_metric=True
            )
    elif args.logger == 'wandb':
        wandb.init()
        wandb.config.update(args)
        logger = WandbLogger(
            project="IC",
            save_dir=os.path.join(args.resutls_dir, "wandb")
            )


    trainer = Trainer(
        logger=logger,
        gpus=avail_gpus,
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10
        )

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
        x_only=args.x_only,
        double_data_by_sym=args.double_data_by_sym,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad
        )


    trainer.fit(
        Wrapped_Model,
        train,
        val
        )
    trainer.test(Wrapped_Model, test_dl)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--model", default='MLP', type=str,
                        choices=['MLP', 'UNET'])

    parser.add_argument("--x_only", default=False, type=bool)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--double_data_by_sym", default=True, type=bool)
    parser.add_argument("--cached", default=True, type=bool)
    parser.add_argument("--max_samples", default=64, type=int,
                        help='-1 for all')

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='mse', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--logger", default='wandb', type=str, choices=['wandb', 'tb'])
    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=False, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args)
