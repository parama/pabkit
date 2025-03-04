from setuptools import setup, find_packages

setup(
    name="pabkit",
    version="0.1.0",
    author="Parama Pal",
    author_email="parama@vsinc.ai",
    description="Process-Aware Benchmarking (PAB) Toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pabkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
)
