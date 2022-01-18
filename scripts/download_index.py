import argparse
import os.path
import requests

from tqdm import tqdm
from pathlib import Path

SPHERE_URL = "http://dl.fbaipublicfiles.com/sphere"

# dense index constants
SPHERE_DENSE_PARTITIONS = 32
PARTITIONS_FILES = ["buffer.pkl", "cfg.json", "meta.pkl", "index.faiss"]


def download_file(url, file_path, overwrite):
    file_name = url.split("/")[-1]
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))

    if not overwrite and os.path.isfile(file_path):
        current_size = os.path.getsize(file_path)
        if total_size == current_size:
            print(" - Skipping " + file_name + " - already exists.")
            return

    block_size = 1024  # 1 Kibibyte
    t = tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        desc=" - Downloading " + file_name + ": ",
    )
    with open(file_path, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def download_sparse(dest_dir, overwrite):
    raise NotImplementedError


def download_dense(dest_dir, overwrite, partitions):
    Path(dest_dir + "/dense").mkdir(parents=True, exist_ok=True)
    for i in range(partitions):
        print("Downloading files for node {} our of {}:".format(i, partitions))
        dense_suffix = "/dense/" + str(i) + "/"
        partition_dir = dest_dir + dense_suffix
        Path(partition_dir).mkdir(exist_ok=True)
        for file_name in PARTITIONS_FILES:
            download_file(
                SPHERE_URL + dense_suffix + file_name, partition_dir + file_name, overwrite
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest_dir",
        required=True,
        type=str,
        help="The path to a directory where index files should be stored",
    )
    parser.add_argument(
        "--index_type",
        required=True,
        choices=["dense", "sparse"],
        type=str,
        help="The type of index to download, choose dense or sparse.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If flag set, existing files will be overwritter, otherwise skipping download.",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=SPHERE_DENSE_PARTITIONS,
        help="The number of partitions the dense index is split into.",
    )
    args = parser.parse_args()

    if args.index_type == "dense":
        download_dense(args.dest_dir, args.overwrite, args.partitions)
    elif args.index_type == "sparse":
        download_sparse(args.dest_dir, args.overwrite)
    else:
        raise ValueError

