from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pab-toolkit",
    version="0.1.0",
    author="PAB Team",
    author_email="info@pab-toolkit.org",
    description="Process-Aware Benchmarking (PAB) Toolkit for Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pab-team/pab-toolkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
)
