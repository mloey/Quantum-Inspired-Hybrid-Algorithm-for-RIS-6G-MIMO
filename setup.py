"""
Setup configuration for QI-HFPA-DRL package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qi-hfpa-drl",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Quantum-Inspired Hybrid Flamingo-Pangolin Algorithm with Deep RL for 6G MIMO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mloey/Quantum-Inspired-Hybrid-Algorithm-for-RIS-6G-MIMO",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
)
