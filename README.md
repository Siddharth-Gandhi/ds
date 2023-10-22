# Neural Code Retrieval for Github Issues/Feature Requests

The main goal of this project is:
> Recent code language models have demonstrated strong capabilities in generating, completing, editing and debugging code. Given certain prompts, they are able to provide human-quality completions. A very capable code language model, however, only captures part of the entire programming workflow. In the case of improving a big code project, after designing some new features, the first thing the programmer would consider is to locate the existing code files to be edited. This is a retrieval problem and is non-trivial in large code bases.
>
> In this project, we’d like to investigate this retrieval problem. Specifically, we want to tackle the feature request/design doc to code files retrieval problem using the state-of-the-art neural retrieval techniques. We hope to mine the training dataset from open source projects, create test cases with human filtering, train neural retrieval models and create retrieval systems. We’d like to produce datasets, open source models, a research paper and a live demo at the end of the project.

## Setup

This project has been setup in Python 3.9.12. First, create a conda environment with

```bash
conda create -n ds python=3.9.12
conda activate ds
```

To install the dependencies, run

```bash
pip install -r requirements.txt
```

## Installing Pyserini

First, install pytorch and faiss, which is very platform dependent. For M1 Mac, use

```
conda install -c conda-forge faiss-cpu
pip3 install torch torchvision torchaudio
```

For an ARM based Mac (M1/M2), use these commands to install pyserini (Python 3.8+)

```
conda install -c conda-forge openjdk=11
CFLAGS="-mavx -DWARN(a)=(a)" pip install nmslib
conda install --yes -c conda-forge 'lightgbm>=3.3.3'
pip install pyserini
```
