import numpy as np
import h5py as h5
from tqdm import tqdm 
import horovod.tensorflow.keras as hvd
import os
import sys

def rotate_data(polar_data,shift):
    # Polar data with dimensions (-1, 45, 16, 9) --> (num_samples, z, alpha, r)

    # Perform the cyclic permutation along the theta dimension
    cyclic_permuted_data = np.roll(polar_data, shift=shift, axis=2)

    return cyclic_permuted_data

if __name__ == "__main__":
    hvd.init()    
    file_path = '/share/lustre/tmadula/External/Data/CaloChallenge2022/dataset_2_1.hdf5'
    file_out = '/share/lustre/tmadula/External/Data/CaloChallenge2022/dataset_2_1_augmented.hdf5'
    
    h5f_ = h5.File(file_path, 'r')
    e = h5f_['incident_energies']
    showers = h5f_['showers']
    
    print(f"Original Shower Shape: {showers.shape}")

    shape = (-1,45,16,9)
    showers = np.reshape(showers, shape)

    print(f"Reshaped Shower Shape: {showers.shape}")

    aug_data = []
    aug_e = np.repeat(e, repeats=16, axis=0)
    for i in range(shape[2]):
        showers_ = rotate_data(showers,i)
        aug_data.append(showers_)
    
    n_cycles = len(aug_data)
    n_showers = len(showers)
    aug_data_ = np.zeros((n_cycles * n_showers, 45, 16, 9))
    for i in range(n_cycles):
        aug_data_[i*n_showers:(i+1)*n_showers] = aug_data[i]
    
    aug_data_ = aug_data_.reshape((aug_data_.shape[0], -1))

    print(f"Shape of final shower dataset: {aug_data_.shape}")
    print(f"Shape of final energy dataset: {aug_e.shape}")
    with h5.File(file_out,"w") as h5f:
        dset = h5f.create_dataset("showers", data=aug_data_, compression="gzip")
        dset = h5f.create_dataset("incident_energies", data=aug_e, compression="gzip")
    
    h5f_.close()