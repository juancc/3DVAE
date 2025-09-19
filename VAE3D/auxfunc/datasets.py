"""
Dataset auxliary functions

"""



import os
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf


def pad_to_cube(voxel, target_shape=(32, 32, 32)):
    """
    Pad a voxel grid with zeros to fit into a fixed cube.
    """
    padded = np.zeros(target_shape, dtype=np.float32)
    min_shape = np.minimum(voxel.shape, target_shape)

    # Center the voxel object
    offset = [(t - s) // 2 for t, s in zip(target_shape, min_shape)]
    slices_in = tuple(slice(0, s) for s in min_shape)
    slices_out = tuple(slice(o, o + s) for o, s in zip(offset, min_shape))

    padded[slices_out] = voxel[slices_in]
    return padded

def load_voxel_dataset(npy_dir, target_shape=(32, 32, 32), test_size=0.2, random_state=42):
    """
    Load voxel .npy files, pad them into a fixed cube, and return TF datasets.
    """
    voxel_files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith(".npy")]
    voxel_files.sort()

    data = []
    for f in voxel_files:
        vox = np.load(f).astype(np.float32)
        vox_padded = pad_to_cube(vox, target_shape)
        data.append(vox_padded)

    data = np.stack(data, axis=0)
    # data = data[..., np.newaxis]  # add channel dimension

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    train_ds = tf.data.Dataset.from_tensor_slices(train_data)
    test_ds = tf.data.Dataset.from_tensor_slices(test_data)

    return train_ds, test_ds
