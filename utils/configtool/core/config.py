# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

"""Configuration file (powered by YACS)."""

import os
import sys
import argparse

from yacs.config import CfgNode


# Global config object
_C = CfgNode(new_allowed=True)
cfg = _C


# -------------------------------------------------------- #
# Data Loader options
# -------------------------------------------------------- #
# _C.CIFAR10_LOADER = CfgNode(new_allowed=True)

# batch size of training and validation
# type: int or list(different during validation)

# ------------------------------------------------------------------------------------ #
# Optimizer options in network
# ------------------------------------------------------------------------------------ #
# _C.TEST = CfgNode(new_allowed=True)


# -------------------------------------------------------- #
# Misc options
# -------------------------------------------------------- #
_C.USE_GPU = True
_C.OUT_DIR = "exp/example"

# -------------------------------------------------------- #

def dump_cfgfile(cfg_dest="config.yaml"):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, cfg_dest)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfgfile(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_configs():
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """
    parser = argparse.ArgumentParser(description="Config file options.")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg)
    _C.merge_from_list(args.opts)
