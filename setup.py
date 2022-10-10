from setuptools import setup, find_packages

setup(
    name='phase2vec',
    description='Learning representations of dynamical systems',
    version='0.1.0',
    packages=find_packages(),
    #platforms=['mac', 'unix'],
    python_requires='>=3.6',
    install_requires=[
         'numpy==1.21.3',
         'torch==1.12.0',
         'matplotlib==3.4.3',
         'tqdm==4.62.3',
         'scikit_learn==1.0.2',
         'scipy==1.7.1',
         'pillow==9.2.0',
         'lightning-bolts',
         'torchvision',
         'torchdiffeq',
         'numba',
         'pandas',
         'typing',
         'click',
         'ruamel.yaml',
    ],
    entry_points={'console_scripts': ['phase2vec = src.cli:cli']}
)
