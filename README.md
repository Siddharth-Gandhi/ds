# Github neural IR code gen thing

- merge requests not present locally
- Python 3.9.12

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
