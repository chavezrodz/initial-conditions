from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning import utilities
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from Datamodule import DataModule
from utils import make_file_prefix
from load_model import load_model
from memory_profiler import profile
import os


# @profile
def main(args):
    utilities.seed.seed_everything(seed=args.seed, workers=True)
    save_path = os.path.join(args.results_dir, 'saved_models', args.train_res, args.train_energy)
    filename = make_file_prefix(args.model_type, args.n_layers, args.hidden_dim, args.kernel_size)
    dm = DataModule(
        datapath=args.datapath,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        res=args.train_res,
        energy=args.train_energy,
        cached=args.cached
        )

    logger = TensorBoardLogger(
        save_dir=os.path.join(save_path, "tb_logs"),
        name=filename,
        default_hp_metric=True
        )
    logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        save_top_k=1,
        monitor='validation/'+args.criterion,
        mode="min",
        filename=filename+'_val_err={validation/sq_err:.2e}',
        auto_insert_metric_name=False,
        save_last=False
        )

    trainer = Trainer(
        logger=logger,
        accelerator='auto',
        devices='auto',
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        auto_lr_find=True,
        strategy="ddp_find_unused_parameters_false",
        )

    model = load_model(
        model_type=args.model_type,
        h_dim=args.hidden_dim,
        n_layers=args.n_layers,
        k_size=args.kernel_size,
        dm=dm,
        saved=False,
        results_dir=args.results_dir,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad
        )

    trainer.fit(
        model,
        datamodule=dm,
        )

    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--model_type", default='MLP', type=str,
                        choices=['MLP', 'UNET'])

    # data params
    parser.add_argument("--max_samples", default=-1, type=int)
    parser.add_argument("--train_res", default='128x128', type=str, choices=['128x128', '512x512'])
    parser.add_argument("--train_energy", default='193', type=str,
                        choices=['193', '2760', '5020', 'all'])

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--cached", default=True, type=bool)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)
    parser.add_argument("--criterion", default='sq_err', type=str,
                        choices=['sum_err', 'abs_err', 'sq_err'])
    parser.add_argument("--add_sum_err", default=True, type=bool)

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    args = parser.parse_args()

    main(args)
