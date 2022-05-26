import os
from argparse import ArgumentParser
from dataloaders import get_iterators
from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.loggers import TensorBoardLogger
from Model import Model
from Wrapper import Wrapper

def main(args):
    utilities.seed.seed_everything(seed=args.seed, workers=True)

    if args.gpu:
        avail_gpus = 1
        n_workers = 0
    else:
        avail_gpus = 0
        n_workers = 8

    train_loader = get_iterators(
        datapath='data',
        cached=args.cached,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        n_workers=n_workers
        )

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.results_dir, "TB_logs"),
        default_hp_metric=True
    )

    trainer = Trainer(
        logger=logger,
        gpus=avail_gpus,
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run
        )

    model = Model(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        )

    Wrapped_Model = Wrapper(
        model,
        x_only=args.x_only,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad
        )


    trainer.fit(
        Wrapped_Model,
        train_loader,
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--x_only", default=True, type=bool)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--input_dim", default=16, type=int)
    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--output_dim", default=16, type=int)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_samples", default=32, type=int)
    parser.add_argument("--cached", default=True, type=bool)

    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='mse', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=False, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args)