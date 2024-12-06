from setuptools import setup, find_packages

setup(
    name="mnist-cnn-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'pytest>=6.0.0',
        'numpy>=1.19.0',
    ],
    python_requires='>=3.8',
) 