from setuptools import setup

setup(
    name='hunayn',
    version='0.1',
    description='Hunayn a Protein/English Transformer translation model',
    packages=['hunayn'],
    package_dir={
        'hunayn': 'hunayn'
    },
    install_requires=[
        'torch>=2.1.0',
        'pytorch-lightning>=2.1.1',
        'torchmetrics>=1.2.0',
        'transformers>=4.0.0',
        'tokenizers>=0.14.0',
        'xformers==0.0.22.post7',
        'wandb>=0.16.0',
        'pandas>=2.0.0',
        'pydantic>=2.4.2',
    ]
)
