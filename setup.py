from setuptools import setup, find_packages

setup(
    name="ring_attn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "flash-attn",
    ],
)
