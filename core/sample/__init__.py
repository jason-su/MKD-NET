from .cornernet import cornernet

def data_sampling_func(sys_configs, db, k_ind, data_aug=True, debug=False):
    return globals()[sys_configs.sampling_function](sys_configs, db, k_ind, data_aug, debug)
