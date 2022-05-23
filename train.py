import os
from argparse import ArgumentParser
from dataloaders import get_iterators
from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.loggers import TensorBoardLogger
from Model import Model
from Wrapper import Wrapper

def main(args):
    train_loader = get_iterators(
        datapath='data',
        batch_size=args.batch_size,
        n_workers=8)

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.results_dir, "TB_logs"),
        default_hp_metric=True
    )

    trainer = Trainer(
        logger=logger,
        gpus=args.avail_gpus,
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
    parser.add_argument("--model", default='lstm', type=str, choices=['gru', 'mlp', 'lstm'])
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--input_dim", default=16, type=int)
    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--output_dim", default=16, type=int)

    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='mse', type=str,
                        choices=['pc_err', 'abs_err', 'mse'])

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_workers", default=8, type=int)
    parser.add_argument("--avail_gpus", default=0, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args)