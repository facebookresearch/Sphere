# Sphere

## Installation
```
git clone git@github.com:facebookresearch/Sphere.git
cd Sphere
pip install -e .
```

## Index download
```bash
python scripts/download_index.py \
    --dest_dir ... \ # The path to a directory where index files should be stored.
    --index_type ... \ # The type of index to download, choose dense or sparse.
    --overwrite \ # If flag set, existing files will be overwritter, otherwise skipping download.
    --partitions ... \ # Optional argument. The number of partitions the dense index is split into.
```
