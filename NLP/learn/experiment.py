
import atexit
import logging
from pathlib import Path

import torch

from NLP.config.factory import (
    DataReaderFactory,
    DataLoaderFactory,
    ModelFactory,
    OptimizerFactory,
)

from NLP import utils as common_utils
from NLP.config.utils import convert_config2dict, pretty_json_dumps, set_global_seed
from NLP.learn.mode import Mode
from NLP.learn.trainer import Trainer
from NLP.learn import utils


logger = logging.getLogger(__name__)


class Experiment:
    """
    Experiment settings with config.

    * Args:
        mode: Mode (ex. TRAIN, EVAL)
        config: (NestedNamespace) Argument config according to mode
    """
    def __init__(self, mode, config):
        common_utils.set_logging_config(mode, config)

        self.argument = (
            config
        )  # self.config (experiment overall config) / config (argument according to mode)
        self.config = config
        self.mode = mode

        self.common_setting(config)

        self.predict_settings = None

    def common_setting(self, config):
        """ Common Setting - experiment config, use_gpu and cuda_device_ids """
        self.config_dict = convert_config2dict(config)

        cuda_devices = self._get_cuda_devices()
        self.config.cuda_devices = cuda_devices
        self.config.slack_url = getattr(self.config, "slack_url", False)

    def _get_cuda_devices(self):
        if getattr(self.config, "use_gpu", None) is None:
            self.config.use_gpu = torch.cuda.is_available()

        if self.config.use_gpu:
            return self.config.cuda_devices
        else:
            return None

    def __call__(self):
        """ Run Trainer """

        set_global_seed(self.config.seed_num)  # For Reproducible

        if self.mode == Mode.TRAIN:
            # exit trigger slack notification
            if self.config.slack_url:
                atexit.register(utils.send_message_to_slack)

            train_loader, valid_loader, optimizer = self.set_train_mode()

            assert train_loader is not None
            assert optimizer is not None

            self.trainer.train(train_loader, optimizer)
            self._summary_experiments()
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def set_train_mode(self):
        """
        Training Mode

        - Pipeline
          1. read raw_data (DataReader)
          2. convert to DataSet (DataReader)
          3. create DataLoader (DataLoader)
          4. define model and optimizer
          5. run!
        """
        logger.info("Config. \n" + pretty_json_dumps(self.config_dict) + "\n")

        data_reader = self._create_data()
        datas, helpers = data_reader.read()

        # iterator
        datasets = data_reader.convert_to_dataset(datas, helpers=helpers)  # with name

        # !!!!! need to make input to train and valid : preprocessing
        self.config.iterator.cuda_devices = self.config.cuda_devices
        train_loader, valid_loader = self._create_by_factory(
            DataLoaderFactory, self.config.iterator, param={"datasets": datasets}
        )

        # calculate 'num_train_steps'
        num_train_steps = self._get_num_train_steps(train_loader)
        self.config.optimizer.num_train_steps = num_train_steps

        checkpoint_dir = Path(self.config.trainer.log_dir) / "checkpoint"
        checkpoints = None
        if checkpoint_dir.exists():
            checkpoints = self._load_exist_checkpoints(checkpoint_dir)  # contain model and optimizer

        if checkpoints is None:
            model = self._create_model(helpers=helpers)
            op_dict = self._create_by_factory(
                OptimizerFactory, self.config.optimizer, param={"model": model}
            )
        else:
            model = self._create_model(checkpoint=checkpoints)
            op_dict = self._create_by_factory(
                OptimizerFactory, self.config.optimizer, param={"model": model}
            )
            utils.load_optimizer_checkpoint(op_dict["optimizer"], checkpoints)

        self.set_trainer(model, op_dict=op_dict)
        return train_loader, valid_loader, op_dict["optimizer"]

    def _create_data(self):
        # !!!! python preprocessing.py --in_dir ljspeech --out_dir DATASETS/ljspeech
        data_reader = self._create_by_factory(DataReaderFactory, self.config.data_reader)
        return data_reader

    def _create_by_factory(self, factory, item_config, param={}):
        return factory(item_config).create(**param)

    def _get_num_train_steps(self, train_loader):
        train_set_size = len(train_loader.dataset)
        batch_size = self.config.iterator.batch_size
        gradient_accumulation_steps = getattr(self.config.optimizer, "gradient_accumulation_steps", 1)
        num_epochs = self.config.trainer.num_epochs

        one_epoch_steps = int(train_set_size / batch_size / gradient_accumulation_steps)
        if one_epoch_steps == 0:
            one_epoch_steps = 1
        num_train_steps = one_epoch_steps * num_epochs
        return num_train_steps

    def _load_exist_checkpoints(self, checkpoint_dir):  # pragma: no cover
        checkpoints = utils.get_sorted_path(checkpoint_dir, both_exist=True)

        train_counts = list(checkpoints.keys())
        if not train_counts:
            return None

        seperator = "-" * 50
        message = f"{seperator}\n !! Find exist checkpoints {train_counts}.\n If you want to recover, input train_count in list.\n If you don't want to recover, input 0.\n{seperator}"
        selected_train_count = common_utils.get_user_input(message)

        if selected_train_count == 0:
            return None

        model_path = checkpoints[selected_train_count]["model"]
        model_checkpoint = self._read_checkpoint(self.config.cuda_devices, model_path)

        optimizer_path = checkpoints[selected_train_count]["optimizer"]
        optimizer_checkpoint = self._read_checkpoint("cpu", optimizer_path)

        checkpoints = {}
        checkpoints.update(model_checkpoint)
        checkpoints.update(optimizer_checkpoint)
        return checkpoints

    def _create_model(self, checkpoint=None, helpers=None):
        if checkpoint is None:
            assert helpers is not None
            first_key = next(iter(helpers))
            helper = helpers[first_key]  # get first helper
            model_init_params = helper.get("model", {})
            predict_helper = helper.get("predict_helper", {})
        else:
            model_init_params = checkpoint.get("init_params", {})
            predict_helper = checkpoint.get("predict_helper", {})

        model_params = {}  # ??????
        model_params.update(model_init_params)

        model = self._create_by_factory(
            ModelFactory, self.config.model, param=model_params
        )
        # Save params
        model.init_params = model_init_params
        model.predict_helper = predict_helper

        if checkpoint is not None:
            model = utils.load_model_checkpoint(model, checkpoint)
        model = self._set_gpu_env(model)
        return model

    def _set_gpu_env(self, model):
        if self.config.use_gpu:
            cuda_devices = self._get_cuda_devices()
            num_gpu = len(cuda_devices)

            use_multi_gpu = num_gpu > 1
            if use_multi_gpu:
                model = torch.nn.DataParallel(model, device_ids=cuda_devices)
            model.cuda()
        else:
            num_gpu = 0

        num_gpu_state = num_gpu
        if num_gpu > 1:
            num_gpu_state += " (Multi-GPU)"
        logger.info(f"use_gpu: {self.config.use_gpu} num_gpu: {num_gpu_state}, distributed training: False, 16-bits trainiing: False")
        return model

    def set_trainer(self, model, op_dict={}, save_params={}):
        trainer_config = vars(self.config.trainer)
        trainer_config["config"] = self.config_dict
        trainer_config["model"] = model
        trainer_config["learning_rate_scheduler"] = op_dict.get("learning_rate_scheduler", None)
        trainer_config["exponential_moving_average"] = op_dict.get(
            "exponential_moving_average", None
        )
        self.trainer = Trainer(**trainer_config)

    def _read_checkpoint(self, cuda_devices, checkpoint_path, prev_cuda_device_id=None):
        if cuda_devices == "cpu":
            return torch.load(checkpoint_path, map_location="cpu")  # use CPU

        if torch.cuda.is_available():
            checkpoint = torch.load(
                checkpoint_path,
                map_location={
                    f"cuda:{prev_cuda_device_id}": f"cuda:{cuda_devices[0]}"
                },  # different cuda_device id case (save/load)
            )
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")  # use CPU
        return checkpoint

    def _summary_experiments(self):
        hr_text = "-" * 50
        summary_logs = f"Config.\n{pretty_json_dumps(self.config_dict)}\n{hr_text}\n"
        summary_logs += (
            f"Training Logs.\n{pretty_json_dumps(self.trainer.training_logs)}\n{hr_text}\n"
        )
        summary_logs += f"Metric Logs.\n{pretty_json_dumps(self.trainer.metric_logs)}"

        logger.info(summary_logs)

        if self.config.slack_url:  # pragma: no cover
            simple_summary_title = f"Session Name"
            if getattr(self.config, "base_config", None):
                simple_summary_title += f"({self.config.base_config})"

            simple_summary_logs = f" - Dataset: {self.config.data_reader.dataset} \n"
            simple_summary_logs += f" - Model: {self.config.model.name}"

            best_metrics = {"epoch": self.trainer.metric_logs["best_epoch"]}
            best_metrics.update(self.trainer.metric_logs["best"])

            simple_summary_logs += f" - Best metrics.\n {pretty_json_dumps(best_metrics)} "

            utils.send_message_to_slack(self.config.slack_url, title=simple_summary_title, message=simple_summary_logs)
