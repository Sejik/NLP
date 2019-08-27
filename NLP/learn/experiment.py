
from pathlib import Path

import torch

from NLP.config.factory import (
    DataReaderFactory,
    DataLoaderFactory,
    ModelFactory,
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
        # TODO


    def common_setting(self, config):
        """ Common Setting - experiment config, use_gpu and cuda_device_ids """
        self.config_dict = convert_config2dict(config)

        cuda_devices = self._get_cuda_devices()
        self.config.cuda_devices = cuda_devices

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
        datas, helpers = data_reader.read()

        datasets = data_reader.convert_to_dataset(datas, helpers=helpers)

        train_loader, synthesize_loader = self._create_by_factory(
            DataLoaderFactory, self.config, param={"datasets": datasets}
        )

        num_train_steps, num_synthesize_steps = self._get_num_train_steps(train_loader)
        self.config.general.num_train_steps = num_train_steps
        self.config.general.num_synthesize_steps = num_synthesize_steps

        checkpoint_dir = Path(self.config.general.log_dir) / "checkpoint"
        checkpoints = None
        # TODO: CHECKPOINT_DIR.EXISTS():

        if checkpoints is None:
            model = self._create_model(helpers=helpers)
            # TODO
        # TODO: else

        # self.set_trainer(model, op_dict=op_dict)
        # return train_loader, op_dict["optimzier"]

    def _create_data(self):
        data_reader = self._create_by_factory(DataReaderFactory, self.config.general)
        return data_reader

    def _create_by_factory(self, factory, item_config, param={}):
        return factory(item_config).create(**param)

    def _get_num_train_steps(self, train_loader):
        set_size = len(train_loader.dataset)
        train_batch_size = self.config.train_batch_size
        synthesize_batch_size = self.config.synthesize_batch_size
        num_epochs = self.config.general.num_epochs

        train_one_epoch_steps = int(set_size / train_batch_size)
        if train_one_epoch_steps == 0:
            train_one_epoch_steps = 1
        num_train_steps = train_one_epoch_steps * num_epochs

        synthesize_one_epoch_steps = int(set_size / synthesize_batch_size)
        if synthesize_one_epoch_steps == 0:
            synthesize_one_epoch_steps = 1
        num_synthesize_steps = synthesize_one_epoch_steps * num_epochs

        return num_train_steps, num_synthesize_steps

    def _create_model(self, checkpoint=None, helpers=None):
        if checkpoint is None:
            assert helpers is not None
            first_key = next(iter(helpers))
            helper = helpers[first_key]  # get first helper
            model_init_params = helper.get("model", {})
            predict_helper = helper.get("predict_helper", {})
        # TODO: else

        model = self._create_by_factory(  # 파라미터 확인
            ModelFactory, self.config.general
        )
        model.init_params = model_init_params
        model.predict_helper = predict_helper

        # TODO: checkpoint
        return model
