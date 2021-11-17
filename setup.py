from setuptools import setup, find_packages

setup(
    name="sealrtc",
    author='Aditya R. Sengupta',
    version="1.1",
    packages=find_packages(where="sealrtc"),
    package_dir={"": "sealrtc"}
)