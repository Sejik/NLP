"""
import atexit
import logging
from pathlib import Path

import torch

from claf import nsml
from claf.config.factory import (
    DataReaderFactory,
    DataLoaderFactory,
    TokenMakersFactory,
    ModelFactory,ƒ
    OptimizerFactory,
)
from claf import utils as common_utils
from claf.config.args import NestedNamespace
from claf.config.utils import convert_config2dict, pretty_json_dumps, set_global_seed
from claf.tokens.text_handler import TextHandler
from claf.learn.mode import Mode
from claf.learn.trainer import Trainer
from claf.learn import utils
"""

from claf.config.utils import set_global_seed
from claf.learn.mode import Mode

class Experiment:
    """

    """

    def __init__(self):

    def __call__(self):
        """Run Trainer"""

        set_global_seed(self.config.seed_num) # For Reproducible

        if self.mode == Mode.TRAIN:
            # exit trigger slack notification
            if self.config.slack_url:
                atexit.register(utils.send_message_to_slack)




# from claf.config import args
# from claf.learn.experiment import Experiment
# from claf.learn.mode import Mode

# experiment = Experiment(Mode.TRAIN, args.config(mode=Mode.TRAIN))
# experiment()


# from claf.config import args
# from claf.learn.experiment import Experiment
# from claf.learn.mode import Mode

# experiment = Experiment(Mode.TRAIN, args.config(mode=Mode.TRAIN))
# experiment()
