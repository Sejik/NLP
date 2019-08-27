
import torch

from NLP.config.factory import (
    DataReaderFactory,
    DataLoaderFactory,
    ModelFactory,
    OptimizerFactory,
)

from NLP.config.utils import convert_config2dict
from NLP.learn.trainer import Trainer


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

        train_loader, synthesize_loader, teacher_optimizer, student_optimizer = self.set_train_mode()
        # TODO
        assert train_loader is not None
        assert synthesize_loader is not None
        assert teacher_optimizer is not None
        assert student_optimizer is not None

        self.teacher_trainer.train(train_loader, teacher_optimizer)
        self.student_trainer.train(train_loader, student_optimizer)
        # self._summary_experiments()
        # valid_loader = self.set_synthesize_mode()
        # assert valid_loader is not None
        # valid_loader = self


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

        # checkpoint_dir = Path(self.config.general.log_dir) / "checkpoint"
        checkpoints = None
        # TODO: CHECKPOINT_DIR.EXISTS():

        if checkpoints is None:
            teacher_model, student_model = self._create_model(helpers=helpers)
            teacher_op_dict = self._create_by_factory(
                OptimizerFactory, self.config.optimizer, param={"model": teacher_model}
            )
            student_op_dict = self._create_by_factory(
                OptimizerFactory, self.config.optimizer, param={"model": student_model}
            )
            # TODO
        # TODO: else

        # TODO: set trainer with teacher and student and run two
        # future update to function in function
        self.set_trainer(teacher_model, student_model, teacher_op_dict=teacher_op_dict, student_op_dict=student_op_dict)
        return train_loader, synthesize_loader, teacher_op_dict["optimizer"], student_op_dict["optimizer"]

    def _create_data(self):
        data_reader = self._create_by_factory(DataReaderFactory, self.config.general)
        return data_reader

    def _create_by_factory(self, factory, item_config, param={}):
        return factory(item_config).create(**param)

    def _get_num_train_steps(self, train_loader):
        set_size = len(train_loader.dataset)
        train_batch_size = self.config.train_batch_size
        synthesize_batch_size = self.config.synthesize_batch_size
        num_epochs = self.config.trainer.num_epochs

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
        # TODO: checkpoint
        assert helpers is not None
        first_key = next(iter(helpers))
        helper = helpers[first_key]  # get first helper
        model_init_params = helper.get("model", {})
        predict_helper = helper.get("predict_helper", {})

        model_params = {}  # TODO

        # TODO: have to seperate onn way model
        teacher_model, student_model = self._create_by_factory(  # 파라미터 확인
            ModelFactory, self.config.general, param=model_params
        )
        teacher_model.init_params = model_init_params
        teacher_model.predict_helper = predict_helper
        student_model.init_params = model_init_params
        student_model.predict_helper = predict_helper

        # TODO: checkpoint
        return teacher_model, student_model

    def set_trainer(self, teacher_model, student_model, teacher_op_dict={}, student_op_dict={}, teacher_save_params={}, student_save_params={}):
        teacher_trainer_config = vars(self.config.trainer)
        teacher_trainer_config["config"] = self.config_dict
        teacher_trainer_config["model"] = teacher_model
        teacher_trainer_config["learning_rate_scheduler"] = teacher_op_dict.get("learning_rate_scheduler", None)
        teacher_trainer_config["exponential_moving_average"] = teacher_op_dict.get(
            "exponential_moving_average", None
        )
        self.teacher_trainer = Trainer(**teacher_trainer_config)
        student_trainer_config = vars(self.config.trainer)
        student_trainer_config["config"] = self.config_dict
        student_trainer_config["model"] = student_model
        student_trainer_config["learning_rate_scheduler"] = student_op_dict.get("learning_rate_scheduler", None)
        student_trainer_config["exponential_moving_average"] = student_op_dict.get(
            "exponential_moving_average", None
        )
        self.student_trainer = Trainer(**student_trainer_config)
