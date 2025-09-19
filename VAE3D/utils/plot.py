"""
Plotting auxiliary functions

"""
import numpy as np
import matplotlib.pyplot as plt

from VAE3D.utils.models3d import save_as_pointcloud



def generate_and_plot(n_samples, model, threshold=0.5, grid_size=None, savepath=None):
    """
    Generate and plot voxel grids in a mosaic layout from a model's latent space.

    Args:
        n_samples (int): number of voxel grids to generate
        model (keras.Model): trained generative model
        threshold (float): cutoff value for binarizing voxel output
        grid_size (tuple): (rows, cols) for mosaic grid.
                           If None, it tries to make a square grid.
    """
    # Default grid layout
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_samples)))
        rows = int(np.ceil(n_samples / cols))
    else:
        rows, cols = grid_size

    n = 4
    new_voxels = model.sample(n_samples=n_samples)

    # Plot mosaic
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                             subplot_kw={"projection": "3d"})
    axes = np.array(axes).reshape(-1)  # flatten in case of single row/col

    for i in range(rows * cols):
        ax = axes[i]
        ax.set_axis_off()
        if i < n_samples:
            vox_out = np.squeeze(new_voxels[i]) > threshold

            # Save point cloud
            if savepath:
                save_as_pointcloud(vox_out, savepath, i, format='ply', voxel_size=1.0, name='points')


            ax.voxels(vox_out, edgecolor="k")
        else:
            ax.set_visible(False)  # hide unused subplots

    plt.tight_layout()
    if savepath:
         plt.savefig(f'{savepath}/mosaic.pdf')

    plt.show()


def plot_reconstruction(dataset, model, n=4):
    # Take one batch
    x_batch = next(iter(dataset.take(1)))
    reconstructions = model.reconstruct(x_batch)

    # Number of samples to visualize
    x_batch = x_batch.numpy()

    fig = plt.figure(figsize=(8, 4 * n))

    for i in range(n):
        # --- Original ---
        ax = fig.add_subplot(n, 2, 2*i + 1, projection='3d')
        vox_in = np.squeeze(x_batch[i]) > 0.5   # make it 3D boolean
        ax.voxels(vox_in, facecolors='blue', edgecolor='k')
        ax.set_title("Input")
        ax.set_axis_off()

        # --- Reconstruction ---
        ax = fig.add_subplot(n, 2, 2*i + 2, projection='3d')
        vox_out = np.squeeze(reconstructions[i]) > 0.5
        ax.voxels(vox_out, facecolors='red', edgecolor='k')
        ax.set_title("Reconstruction")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


    def plot_history(history, figsize=(8, 5)):
        # Plot training & validation accuracy values
        plt.figure(figsize=figsize)

        plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
        plt.plot(history.history['kl_loss'], label='KL Loss')
        plt.plot(history.history['loss'], label='Training Loss')


        # plt.plot(history.history['val_reconstruction_loss'], label='Reconstruction Loss')
        # plt.plot(history.history['val_kl_loss'], label='KL Loss')
        # plt.plot(history.history['val_loss'], label='Training Loss')


        plt.title('Traininf')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()