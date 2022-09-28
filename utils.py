import numpy as np
from MLP import MLP
from UNET import UNET
from Wrapper import Wrapper
import os

def value_to_idx(x):
    unique_values = np.sort(np.unique(x))
    for idx, value in enumerate(reversed(unique_values)):
        x[np.argwhere(x == value)] = len(unique_values) - 1 - idx
    return x.astype(int)

def two_to_three(arr):
    """
    shape: (npts, cols)
    First two columns are xy coords
    """
    arr = arr.copy()
    len_arr, cols  = arr.shape
    n = int(np.sqrt(len_arr))


    # Sanity check, these indices should be range(len_arr)
    arr[:, 0] = value_to_idx(arr[:, 0])
    arr[:, 1] = value_to_idx(arr[:, 1])
    
    idx = np.ravel_multi_index(
        (arr[:, 0].astype(int), arr[:, 1].astype(int)), (n, n)
        )

    assert np.array_equal(idx, np.arange(len_arr))

    arr = np.reshape(arr[:,2:], (n,n, cols - 2))
    return arr




def make_file_prefix(args):
    file_prefix = 'tr_res_' + str(args.train_res)
    file_prefix += '_e_' + str(args.train_energy)
    file_prefix += '_M_' + str(args.model)
    file_prefix += '_nl_' + str(args.n_layers)
    file_prefix += '_hdim_' + str(args.hidden_dim)
    file_prefix += '_ksize_' + str(args.kernel_size)
    return file_prefix

def load_model(args, dm, saved):
    if args.model == 'MLP': 
        model = MLP(
            input_dim=dm.input_dim,
            output_dim=dm.output_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size
            )
    elif args.model == 'UNET':
        model = UNET(
            input_dim=dm.input_dim,
            output_dim=dm.output_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size
            )

    if not saved:
        wrapped_model = Wrapper(
            model,
            dm.norms,
            criterion=args.criterion,
            lr=args.lr,
            amsgrad=args.amsgrad
            )
        return wrapped_model
    else:
        model_file = make_file_prefix(args)+f'_val_err_{args.pc_err}.ckpt'
        model_path = os.path.join(
            args.results_dir, "saved_models", model_file
            )
        wrapped_model = Wrapper.load_from_checkpoint(
            core_model=model,
            norms=dm.norms,
            criterion=args.criterion,
            lr=args.lr,
            amsgrad=args.amsgrad,
            checkpoint_path=model_path,
            )
        return wrapped_model, model_file
