import numpy as np

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

def make_file_prefix(train_res, train_energy, model, n_layers, hidden_dim, kernel_size):
    file_prefix = 'tr_res_' + str(train_res)
    file_prefix += '_e_' + str(train_energy)
    file_prefix += '_M_' + str(model)
    file_prefix += '_nl_' + str(n_layers)
    file_prefix += '_hdim_' + str(hidden_dim)
    file_prefix += '_ksize_' + str(kernel_size)
    return file_prefix

