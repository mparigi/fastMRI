"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import sys
from argparse import ArgumentParser

import h5py
import shutil

sys.path.append("../../")  # noqa: E402


def make_new_dataset(data_dir, out_dir, desired_coils=20):
    for f in data_dir.iterdir():
        with h5py.File(f, "r") as hf:
            num_coils = hf['kspace'].shape[1]
            if num_coils == desired_coils:
                shutil.copyfile(f, out_dir / f.name)


def create_arg_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_path", type=pathlib.Path, required=True, help="Path to the data",
    )
    parser.add_argument(
        "--out_path",
        type=pathlib.Path,
        required=True,
        help="Path to save the reconstructions to",
    )

    parser.add_argument(
        "--desired_coils",
        type=int,
        required=True
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    make_new_dataset(args.data_path, args.out_path, args.desired_coils)
