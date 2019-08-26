
import torch

from NLP.config.factory import (
    DataReaderFactory,
)

from NLP.config.utils import convert_config2dict


class Experiment:
    """
    Experiment settings with config.

    * Args:
        mode: Mode (ex. ALL_IN_ONE)
        config: (NestedNamespace) Argument config according to mode
    """

    def __init__(self, mode, config):
        self.mode = mode
        self.config = config

        self.common_setting(config)

    def __call__(self):
        """ Run Trainer """

        self.set_train_mode()


    def common_setting(self, config):
        """ Common Setting - experiment config, use_gpu and cuda_device_ids """
        self.config_dict = convert_config2dict(config)

        cuda_devices = self._get_cuda_devices()
        self.config.cuda_davices = cuda_devices

    def _get_cuda_devices(self):
        if getattr(self.config, "use_gpu", None) is None:
            self.config.use_gpu = torch.cuda.is_available()

        if self.config.use_gpu:
            return self.config.cuda_devices
        else:
            return None

    def set_train_mode(self):
        """
        Training Mode

        - Pipeline
        1. read raw_data (DataReader)
        """

        data_reader = self._create_data()
        datas, helper = data_reader.read()  # TODO

        # 데이터를 어떻게 넘겨주는 것이 좋을지 생각
        # 데이터를 batch로 어떻게 읽을지까지 가져옴

        # iterator
        # data_loader = self._create_by_factory(DataLoaderFactory, self.config)
        # num_train_steps = self._get_num_train_steps(data_loader)

        # 파라미터 확인
        # checkpoint_dir = Path(self.config.trainer.log_dir) / "checkpoint"
        # checkpoints = None
        # checkpoint exist, None, not non
        # OptimizerFactory

        # set trainer

    def _create_data(self):
        data_reader = self._create_by_factory(DataReaderFactory, self.config.general)
        return data_reader

    def _create_by_factory(self, factory, item_config, param={}):
        return factory(item_config).create(**param)
