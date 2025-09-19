"""
Dataset preprocessing
- Visualizations
- Convert from mesh to voxels

Run with: python -m Scripts.preprocessing

JCA

"""

from VAE3D.utils import models3d as aux3dmodel


DATASET_PATH = '/Users/jarbel16/Downloads/ModelNet10/chair/train/'
OUTPUT_PATH = '/Users/jarbel16/Downloads/Chair'

RESOLUTION = 32

# Number of files to visualizate 
NO_VISUALIZATION = 10

if __name__ == '__main__':
    # # Visualization
    # # Export some models to stl
    # print(' - Exporting to STL some files')
    # aux3dmodel.convert_off_to_stl(DATASET_PATH, OUTPUT_PATH+'/stl', number_files=NO_VISUALIZATION)

    # print(' - Render some Voxels')
    # aux3dmodel.render_voxels(DATASET_PATH, 
    #                        OUTPUT_PATH+'/voxels_render', 
    #                        number_files=NO_VISUALIZATION, 
    #                        resolution=RESOLUTION)
    
    print(' - Exporting dataset to voxels')
    aux3dmodel.export_dataset_voxels(DATASET_PATH, 
                           OUTPUT_PATH+'/voxels', 
                           number_files=None, 
                           resolution=RESOLUTION)