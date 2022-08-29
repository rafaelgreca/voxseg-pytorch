from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="voxseg-pytorch",
    version="0.1.0",
    author="Rafael Greca",
    author_email="rafaelgreca97@hotmail.com",
    description="Voxseg VAD implemented in PyTorch",
    long_description=readme,
    url="https://github.com/rafaelgreca/voxseg-pytorch",
    packages=find_packages(),
    license=license,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    python_requires=">=3.8",
    install_requires=[
        "wheel",
        "pandas",
        "scipy",
        "tables",
        "python_speech_features",
        "tensorflow",
    ],
)
