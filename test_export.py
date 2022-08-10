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


def load_model(args, norms):
    h_dim = args.hidden_dim
    n_layers = args.n_layers
    method = args.model
    results_dir = args.results_dir
    pc_err = args.pc_err

    model_file = f'M_{method}_n_layers_{n_layers}_hid_dim_{h_dim}'
    model_file += f'_{pc_err}.ckpt'
    model_path = os.path.join(
        results_dir, "saved_models", model_file
        )

    input_dim = 8 if args.x_only else 16
    input_dim *= 2 # concatenating A&B
    output_dim = 8 if args.x_only else 16

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
        rm_zeros=args.remove_zero_targets,
        x_only=args.x_only,
        double_data_by_sym=args.double_data_by_sym,
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

    fig,axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
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
        n_workers=1
        )

    model = load_model(args, norms)

    if args.include_test:
        for batch in test_dl:
            pred, y = model.predict_step(batch,0)
            visualize_target_output(pred.detach(), y.detach())


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
    parser.add_argument("--export", default=False, type=bool)
    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--proj_dir", default='rate_integrating', type=str)
    parser.add_argument("--datapath", default='data', type=str)
    parser.add_argument("--which_spacing", default='both', type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--cached", default=True, type=bool)
    parser.add_argument("--x_only", default=False, type=bool)
    parser.add_argument("--remove_zero_targets", default=False, type=bool)
    parser.add_argument("--double_data_by_sym", default=True, type=bool)
    parser.add_argument("--criterion", default='sq_err', type=str,
                        choices=['sum_err', 'abs_err', 'sq_err'])
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)

    # Model Params
    parser.add_argument("--model", default='UNET', type=str)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--pc_err", default='5.13e-01', type=str)

    # Rate Integrating
    args = parser.parse_args()

    main(args)
7