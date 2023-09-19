import numpy as np
from MLP import MLP
from UNET import UNET
from Wrapper import Wrapper
from utils import make_file_prefix
import glob
import os


def load_model(
        model_type, h_dim, n_layers, k_size, dm, saved, results_dir,
        criterion=None, lr=None, amsgrad=None, pc_err=None,
        train_energy=None,train_res=None):
    if model_type == 'MLP': 
        model = MLP(
            input_dim=dm.input_dim,
            output_dim=dm.output_dim,
            hidden_dim=h_dim,
            n_layers=n_layers,
            kernel_size=k_size
            )
    elif model_type == 'UNET':
        model = UNET(
            input_dim=dm.input_dim,
            output_dim=dm.output_dim,
            hidden_dim=h_dim,
            n_layers=n_layers,
            kernel_size=k_size
            )
    if saved:
        save_path = os.path.join(results_dir, 'saved_models', train_res, train_energy)
        model_file = make_file_prefix(model_type, n_layers, h_dim, k_size)
        matching_models = glob.glob(os.path.join(save_path, '*'+model_file+'*'))
        tmp_models = [
            float(match.split('=')[1].strip('.ckpt'))
            for match in matching_models
            ]
        idx = np.argmin(tmp_models)
        print(f"""
        Loading {model_type} nl {n_layers} hdim {h_dim} with {tmp_models[idx]:e} error
             """)
        wrapped_model = Wrapper.load_from_checkpoint(
            core_model=model,
            norms=dm.norms,
            criterion=criterion,
            lr=lr,
            amsgrad=amsgrad,
            checkpoint_path=matching_models[idx],
            )
        return wrapped_model, model_file
    else:
        wrapped_model = Wrapper(
            model,
            dm.norms,
            criterion=criterion,
            lr=lr,
            amsgrad=amsgrad
            )
        return wrapped_model
