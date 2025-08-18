from setuptools import setup, find_packages

setup(
    name="evoaug-tf",
    version="1.0.4",
    packages=find_packages(),
    description="A Python package providing dataloader-based evolution-inspired data augmentations for TensorFlow genomics models.",
    long_description="EvoAug-TF provides a tf.data.Dataset-based approach for applying evolution-inspired augmentations to genomic sequence data, making it easy to integrate with any TensorFlow/Keras model and training pipeline.",
    python_requires=">=3.6",
    install_requires=[
        "tensorflow>=2.11.0",
        "numpy>=1.21.0"
    ],
    author="Yiyang Yu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
