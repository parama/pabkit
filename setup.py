from setuptools import setup, find_packages

setup(
    name="pabkit",
    version="0.1.0",
    author="Parama Pal",
    author_email="parama@vsinc.ai",
    description="Process-Aware Benchmarking (PAB) Toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/parama/pabkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "pyyaml>=5.1.0",
        "typing-extensions>=3.7.4",
    ],
    entry_points={
        'console_scripts': [
            'pab-cli=pab.cli:main',
        ],
    },
    include_package_data=True,
)