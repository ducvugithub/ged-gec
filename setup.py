"""Setup script for Finnish GEC package."""

from setuptools import setup, find_packages

setup(
    name="finnish-gec",
    version="0.1.0",
    description="Finnish Grammatical Error Correction research project",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
