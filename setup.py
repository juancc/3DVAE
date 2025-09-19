from setuptools import setup, find_packages

setup(
    name='VAE3D',
    version='0.0.1',    
    description='3D Vae for voxels generative design',
    url='https://github.com/juancc/3DVAE',
    author='Juan Carlos Arbelaez',
    author_email='jarbel16@eafit.edu.co',
    license= '',
    packages= find_packages(include=['VAE3D', 'VAE3D.*']),
    install_requires=[
                      'tqdm',
                      'numpy',
                      'matplotlib',
                      'numpy-stl',
                      'open3d',
                      'trimesh',
                      'tensorflow',
                      'keras',
                      ],
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)