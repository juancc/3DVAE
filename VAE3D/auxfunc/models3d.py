"""
Auxiliary functions for 3D models

JCA
"""

import os

import scipy.io
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def export_pascal3d(mat_file_path, save_dir, name):
    # Load CAD data
    cad_data = scipy.io.loadmat(mat_file_path)

    cad_models = cad_data[name][0]  # cell array -> list of structs

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Export each CAD model
    for i, model in enumerate(cad_models):
        vertices = model['vertices']
        faces = model['faces'] - 1  # convert MATLAB 1-based indices to 0-based

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        save_path = os.path.join(save_dir, f"model_{i+1}.stl")
        mesh.export(save_path)
        print(f"Saved {save_path}")


def convert_off_to_stl(input_dir, output_dir, number_files=None):
    """ModelNet dataset format."""
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    no = number_files if number_files else len(files)

    for i in tqdm(range(no), total=no):
        file = files[i]
        if file.endswith(".off"):
            path = os.path.join(input_dir, file)

            # Load OFF mesh
            mesh = trimesh.load(path, file_type='off')

            # Save as STL
            save_path = os.path.join(output_dir, file.replace(".off", ".stl"))
            mesh.export(save_path)
            # print(f"Saved {}: {save_path}")
            i += 1



def mesh_save_to_voxels(input_dir, output_dir, resolution=32,file_type='off'):
    """Load mesh file and convert to voxels"""
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(input_dir)
    for file in tqdm(files, total=len(files)):
        vox = mesh_to_voxels(mesh_file, resolution=res)
        np.save("chair_vox.npy", vox)   # save to disk



def mesh_to_voxels(mesh_path, resolution=32,  file_type='off'):
    """
    Convert a 3D mesh to a voxel grid.
    Returns a boolean 3D numpy array (resolution³).
    """
    mesh = trimesh.load(mesh_path, file_type)

    # Voxelize the mesh
    voxels = mesh.voxelized(pitch=mesh.scale / resolution)

    return voxels.matrix.astype(bool)


def render_voxels(input_dir, output_dir, resolution=32, file_type='off', number_files=None):
    """Render some files voxels"""
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    no = number_files if number_files else len(files)


    for i in tqdm(range(no), total=no):
        file = files[i]
        if file.endswith(".off"):
            path = os.path.join(input_dir, file)
            vox = mesh_to_voxels(path, resolution=resolution)

            filepath = os.path.join(output_dir, f'vox{i}.png')
            visualize_voxels(vox, savepath=filepath, show=False)


def export_dataset_voxels(input_dir, output_dir, resolution=32, file_type='off', number_files=None):
    """Export Dataset in voxels numpy format"""
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    no = number_files if number_files else len(files)


    for i in tqdm(range(no), total=no):
        file = files[i]
        if file.endswith(".off"):
            path = os.path.join(input_dir, file)
            vox = mesh_to_voxels(path, resolution=resolution)
            filepath = os.path.join(output_dir, f'vox{i}.npy')
            np.save(filepath, vox)   # save to disk



def visualize_voxels(voxel_matrix, color='blue', show=True, savepath=None):
    """
    Visualize a voxel grid using matplotlib with clean output (no axes, ticks, or grid).
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(voxel_matrix, facecolors=color, edgecolor='k')

    # Turn off axes, ticks, and grid
    ax.set_axis_off()

    if savepath:
        plt.savefig(savepath)


    if show: plt.show()



# Example usage
if __name__ == "__main__":


    # ModelNet
    # input_folder = "/Users/jarbel16/Downloads/ModelNet10/chair/train"   # OFF files path
    # output_folder = "/Users/jarbel16/Downloads/chair-stl"
    # convert_off_to_stl(input_folder, output_folder)

    # Voxels resolution
    # mesh_file = "/Users/jarbel16/Downloads/ModelNet10/chair/train/chair_0004.off"

    # for res in [16,32,64,128]:
    #     vox = mesh_to_voxels(mesh_file, resolution=res)
    #     # np.save("chair_vox.npy", vox)   # save to disk
    #     visualize_voxels(vox, savepath=f'/Users/jarbel16/Downloads/im{res}.png', show=False)# visualize


    # Render some voxels
    res = 32
    for i in range(6):
        i+=1
        mesh_file = f"/Users/jarbel16/Downloads/ModelNet10/chair/train/chair_000{i}.off"
        vox = mesh_to_voxels(mesh_file, resolution=res)
        visualize_voxels(vox, savepath=f'/Users/jarbel16/Downloads/im{i}.png', show=False) 


