from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from dataloaders  import get_iterators
from Model import Model
from UNET import UNET
from Wrapper import Wrapper
from pytorch_lightning import Trainer
from utils import three_to_two, make_file_prefix

def load_model(args, norms):
    results_dir = args.results_dir

    model_file = make_file_prefix(args)+f'_{args.pc_err}.ckpt'
    model_path = os.path.join(
        results_dir, "saved_models", model_file
        )

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

    wrapped_model = Wrapper.load_from_checkpoint(
        core_model=model,
        norms=norms,
        criterion=args.criterion,
        lr=args.lr,
        amsgrad=args.amsgrad,
        checkpoint_path=model_path,
        )

    return wrapped_model

def visualize_target_output(pred, y):
    pred, y = pred[0], y[0]
    pred, y = pred.mean(dim=0), y.mean(dim=0)

    minmin = torch.min(torch.tensor(
        [pred.min().item(), y.min().item()]
        ))

    maxmax = torch.max(torch.tensor(
        [pred.max().item(), y.max().item()]
        ))

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    im = axs[0].imshow(pred, vmin=minmin, vmax=maxmax, cmap='bone')
    im = axs[1].imshow(y, vmin=minmin, vmax=maxmax, cmap='bone')

    axs[0].set_title('Prediction')
    axs[1].set_title('Target')
    fig.tight_layout()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def main(args):
    results_dir = args.results_dir

    (train, val, test_dl), norms = get_iterators(
        datapath=args.datapath,
        cached=args.cached,
        batch_size=args.batch_size,
        n_workers=8
        )

    model = load_model(args, norms)

    if args.visualize:
        # visualize_target_output(pred.detach(), y.detach())
        pass

    if args.include_test:
        trainer = Trainer(logger=False)
        predictions = trainer.predict(model, test_dl)

        outfolder = os.path.join('Results', 'Predictions', args.res, args.energy)
        os.makedirs(outfolder, exist_ok=True)
        
        sample_file = os.path.join('data',args.res, args.energy,'0.dat')
        f = open(sample_file, 'r')
        header = f.readline()
        f.close()
        data_sample = np.loadtxt(sample_file)
        x_values = np.sort(np.unique(data_sample[:, 0]))

        for batch in predictions:
            pred, target, fns = batch
            for idx, file_nb in enumerate(fns):
                outfile = os.path.join(outfolder, 'pred_'+file_nb+'.dat')

                source = np.loadtxt(
                    os.path.join('data',args.res, args.energy,str(file_nb)+'.dat')
                    )
                processed_target = three_to_two(target[idx].permute((1,2,0)), x_values)
                assert np.allclose(source[:, -16:], processed_target[:, -16:])

                arr = three_to_two(pred[idx].permute((1,2,0)), x_values)
                np.savetxt(outfile, arr, delimiter=',', header=header)

    # Exporting
    if args.export:
        # for batch in test_dl:
        #     x = batch[0]
        #     input_example = x[:4]
        #     break

        compiled_path = os.path.join(
            results_dir,
            "compiled_models",
            args.proj_dir,
            )
        os.makedirs(compiled_path, exist_ok=True)

        model.to_torchscript(
            file_path=compiled_path,
            example_inputs=input_example
            )


if __name__ == '__main__':
    parser = ArgumentParser()
    # Managing params
    parser.add_argument("--include_test", default=True, type=bool)
    parser.add_argument("--visualize", default=False, type=bool)
    parser.add_argument("--export", default=False, type=bool)

    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)

    # data params
    parser.add_argument("--res", default='128x128', type=str)
    parser.add_argument("--energy", default='5020', type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--cached", default=True, type=bool)

    # training params
    parser.add_argument("--criterion", default='sq_err', type=str,
                        choices=['sum_err', 'abs_err', 'sq_err'])
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)

    # Model Params
    parser.add_argument("--model", default='UNET', type=str)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--pc_err", default='1.00e+00', type=str)

    # Rate Integrating
    args = parser.parse_args()

    main(args)
7