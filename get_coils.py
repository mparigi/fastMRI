

import pathlib
import sys
from argparse import ArgumentParser

import h5py

from collections import Counter

sys.path.append("../../")  # noqa: E402

import fastmri
from fastmri.data import transforms


def get_coils(data_dir):
    coil_list = []
    for f in data_dir.iterdir():
        with h5py.File(f, "r") as hf:
            coil_list.append(hf["kspace"].shape[1])

    coil_dict = dict(sorted(Counter(coil_list).items()))

    return coil_dict
    


def create_arg_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_path", type=pathlib.Path, required=True, help="Path to the data",
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    coil_dict = get_coils(args.data_path)
    print(coil_dict)
