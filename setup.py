# setup.py
#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Iterative vocoders for audio phase retrieval",
    author="Sewade Ogun",
    author_email="",
    url="https://github.com/ogunlao/iter_vocoder",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["librosa", "numpy"],
    packages=find_packages(),
)
