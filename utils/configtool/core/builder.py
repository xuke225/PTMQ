import os
import torch
import random
import datetime
import numpy as np


# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS
import sys
sys.path.append("../../../")
import utils.logger as logging

import utils.configtool.core.config as config
from utils.configtool.core.config import cfg



logger = logging.get_logger(__name__)


def setup_env():
    """Set up environment for training or testing."""
    # Ensure the output dir exists and save config
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    config.dump_cfgfile()

    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    timeNow = datetime.datetime.now()
    
    logger.info("runing time :{}".format(timeNow.strftime('%Y-%m-%d %H:%M:%S')))
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
  
    device = 'cuda:0'   # TODO: ddp support
    return device
