from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from Datamodule  import DataModule
from pytorch_lightning import Trainer
from load_model import load_model

def three_to_two(array, x_values):
    """
    shape: (x, y, channels)
    """
    # Verifying shape
    assert array.shape[1] == array.shape[0]


    nx, ny, channels = array.shape
    xv, yv = np.meshgrid(range(nx), range(ny))
    coords = np.stack(( yv.flatten(), xv.flatten()), axis=1)

    array = array.reshape(nx*ny, channels)
    array = np.concatenate([coords, array], axis=1)

    array[:, :2] = x_values[array[:, :2].astype(int)]
    return array


def main(args):
    dm_trained = DataModule(
        datapath=args.datapath,
        cached=args.cached,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        res=args.train_res,
        energy=args.train_energy
        )

    model, model_name = load_model(
        model_type=args.model_type,
        h_dim=args.hidden_dim,
        n_layers=args.n_layers,
        k_size=args.kernel_size,
        dm=dm_trained,
        saved=True,
        results_dir=args.results_dir,
        train_energy=args.train_energy,
        train_res=args.train_res
        )
    del dm_trained

    dm_test = DataModule(
        datapath=args.datapath,
        cached=args.cached,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        res=args.train_res,
        energy=args.train_energy)

    dm_test.setup()

    if args.predict:
        outfolder = os.path.join('Results', 'Predictions', model_name, args.test_res, args.test_energy)
        os.makedirs(outfolder, exist_ok=True)

        sample_file = os.path.join('data','raw', args.test_res, args.test_energy,'0.dat')
        f = open(sample_file, 'r')
        header = f.readline()
        f.close()

        data_sample = np.loadtxt(sample_file)
        x_values = np.sort(np.unique(data_sample[:, 0]))

        dl_pred = dm_test.predict_dataloader()
        count = 0
        for batch_idx, batch in enumerate(dl_pred):
            abc, fns = batch
            ab = abc[:, :32]
            pred, y = model.inference_step(batch, batch_idx, double_data_by_sym=False)
            count+=1
            if count > 10:
                break
            if args.save:
                for idx, file_nb in enumerate(fns):
                    # Merge unscaled inputs to the prediction before reverting to the original format
                    arr = torch.cat([ab[idx], pred[idx]], axis=0)
                    arr = three_to_two(arr.permute((1,2,0)).cpu(), x_values)
                    outfile = os.path.join(outfolder, 'pred_'+file_nb+'.dat')
                    np.savetxt(outfile, arr, delimiter=',', header=header)

    # Exporting
    # if args.export:
    #     # for batch in test_dl:
    #     #     x = batch[0]
    #     #     input_example = x[:4]
    #     #     break

    #     compiled_path = os.path.join(
    #         args.results_dir,
    #         "compiled_models",
    #         args.proj_dir,
    #         )
    #     os.makedirs(compiled_path, exist_ok=True)

    #     model.to_torchscript(
    #         file_path=compiled_path,
    #         example_inputs=input_example
    #         )


if __name__ == '__main__':
    parser = ArgumentParser()
    # Managing params
    parser.add_argument("--predict", default=False, type=bool)
    parser.add_argument("--export", default=False, type=bool)

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--results_dir", default='Results', type=str)
    parser.add_argument("--datapath", default='data', type=str)

    # data params
    parser.add_argument("--train_res", default='128x128', type=str)
    parser.add_argument("--train_energy", default='193', type=str)

    parser.add_argument("--test_res", default='128x128', type=str)
    parser.add_argument("--test_energy", default='193', type=str)

    parser.add_argument("--max_samples", default=-1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--cached", default=True, type=bool)

    # training params
    parser.add_argument("--criterion", default='sq_err', type=str,
                        choices=['sum_err', 'abs_err', 'sq_err'])
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--amsgrad", default=True, type=bool)

    # Model Params
    parser.add_argument("--model_type", default='UNET', type=str)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--pc_err", default='1.80e-01', type=str)

    # Rate Integrating
    args = parser.parse_args()

    main(args)
