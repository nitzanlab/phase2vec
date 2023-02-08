from setuptools import setup, find_packages

setup(
    name='phase2vec',
    description='Learning representations of dynamical systems',
    version='0.1.0',
    packages=find_packages(),
    #platforms=['mac', 'unix'],
    python_requires='>=3.6',
    install_requires=[
        'click>=8.1.3',
        'cycler>=0.11.0',
        'decorator>=4.3.0',
        'functorch>=0.2.1',
        'jupyterlab',
        'notebook',
        'lightning-bolts>=0.5.0',
        'Markdown>=3.4.1',
        'MarkupSafe>=2.1.1',
        'matplotlib>=3.4.3',
        'numba>=0.56.2',
        'numpy>=1.21.3',
        'pandas>=1.5.0',
        'Pillow>=9.2.0',
        'pytorch-lightning>=1.7.7',
        'ruamel.yaml>=0.17.21',
        'ruamel.yaml.clib>=0.2.6',
        'scikit-learn>=1.0.2',
        'scipy>=1.7.1',
        'six>=1.16.0',
        'tensorboard>=2.10.1',
        'tensorboard-data-server>=0.6.1',
        'tensorboard-plugin-wit>=1.8.1',
        'torch>=1.12.1',
        'torchdiffeq>=0.2.3',
        'torchmetrics>=0.10.0',
        'torchvision>=0.13.1',
        'tqdm>=4.62.3',
        'typing>=3.7.4.3',
        'typing_extensions>=4.4.0',
        'urllib3>=1.26.12',
        'Werkzeug>=2.2.2',
        'yarl>=1.8.1',
        'zipp>=3.9.0',
        'decorator>=4.3.0',
        'jupyterlab',
        'notebook'
    ],
    entry_points={'console_scripts': ['phase2vec = phase2vec.cli:cli']}
)
