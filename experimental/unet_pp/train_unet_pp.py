"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

sys.path.append("../../")  # noqa: E402

from fastmri.data.mri_data import fetch_dir
from nested_unet_module import NestedUnetModule


def main(args):
    """Main training routine."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    seed_everything(args.seed)
    model = NestedUnetModule(**vars(args))
    
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING OR TEST
    # ------------------------
    if args.mode == "train":
        trainer.fit(model)
    elif args.mode == "test":
        assert args.resume_from_checkpoint is not None
        trainer.test(model)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    path_config = pathlib.Path.cwd() / "fastmri_dirs.yaml"
    knee_path = fetch_dir("knee_path", path_config)
    logdir = fetch_dir("log_path", path_config) / "unet_pp"

    parent_parser = ArgumentParser(add_help=False)

    parser = NestedUnetModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 0
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    config = dict(
        in_chans=1,
        out_chans=1,
        mask_type="random",
        center_fractions=[0.08],
        accelerations=[4],
        resolution=384,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=knee_path,
        challenge="singlecoil",
        exp_dir=logdir,
        exp_name="unet_pp",
        test_split="test",
        batch_size=batch_size,
    )
    parser.set_defaults(**config)

    parser.set_defaults(
        gpus=num_gpus,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
        deterministic=True,
    )

    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
