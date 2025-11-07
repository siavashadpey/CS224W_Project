from setuptools import setup, find_packages

setup(
    name="cs224w-project",
    version="0.1.0",
    description="CS224W Course Project - EGNN Autoencoder",
    author="TBD",
    author_email="TBD",
    packages=find_packages(exclude=["tests*", "checkpoints*", "data*"]),
    python_requires=">=3.9",
    
    install_requires=[
        "torch>=2.5.0", 
        "torch-geometric",
        "numpy",
    ],
    
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "ipython",
        ]
    },
)