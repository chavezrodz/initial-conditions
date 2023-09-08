from MLP import MLP
from UNET import UNET
from Wrapper import Wrapper
import os


def load_model(model, h_dim, n_layers, k_size, dm, saved, results_dir, criterion, lr, amsgrad, pc_err):
    if model == 'MLP': 
        model = MLP(
            input_dim=dm.input_dim,
            output_dim=dm.output_dim,
            hidden_dim=h_dim,
            n_layers=n_layers,
            kernel_size=k_size
            )
    elif model == 'UNET':
        model = UNET(
            input_dim=dm.input_dim,
            output_dim=dm.output_dim,
            hidden_dim=h_dim,
            n_layers=n_layers,
            kernel_size=k_size
            )
    if saved:
        model_file = make_file_prefix(train_res, train_energy, model, n_layers, h_dim, k_size)
        model_file += f'_val_err_{pc_err}.ckpt'
        model_path = os.path.join(results_dir, "saved_models", model_file)
        wrapped_model = Wrapper.load_from_checkpoint(
            core_model=model,
            norms=dm.norms,
            criterion=criterion,
            lr=lr,
            amsgrad=amsgrad,
            checkpoint_path=model_path,
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
