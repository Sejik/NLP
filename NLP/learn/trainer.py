
import os

import torch

from NLP.learn.optimization.exponential_moving_avarage import EMA
from NLP.learn.tensorboard import TensorBoard
from NLP.learn import utils

class Trainer:
    """
    Trainer
    Run experiment

    - train

    * Args:
        config: experiment overall config
        model: Model based on torch.nn.Module

    * Kwargs:
        log_dir: path to directory for save model and other options
        grad_max_norm: Clips gradient norm of an iterable of parameters.
        learning_rate_scheduler: PyTorch's Learning Rate Scheduler.
            (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html)
        exponential_moving_average: the moving averages of all weights of the model are maintained
            with the exponential decay rate of {ema}.
        num_epochs: the number of maximun epochs (Default is 20)
        early_stopping_threshold: the number of early stopping threshold (Default is 10)
        max_eval_examples: print evaluation examples
        metric_key: metric score's control point
        verbose_step_count: print verbose step count (Default is 100)
        eval_and_save_step_count: evaluate valid_dataset then save every n step_count (Default is 'epoch')
    """

    def __init__(
        self,
        model,
        config={},
        log_dir="logs/experiment",
        grad_max_norm=None,
        gradient_accumulation_steps=1,
        learning_rate_scheduler=None,
        exponential_moving_average=None,
        num_epochs=20,
        early_stopping_threshold=10,
        max_eval_examples=5,
        metric_key=None,
        verbose_step_count=100,
        eval_and_save_step_count="epoch",
    ):
        assert metric_key is not None

        # CUDA
        self.use_multi_gpu = type(model) == torch.nn.DataParallel

        if getattr(model, "train_counter", None):
            self.train_counter = model.train_counter
        else:
            self.train_counter = utils.TrainCounter(display_unit=eval_and_save_step_count)

        self.model = model
        model_config = config.get("model", {})
        self.model_name = model_config.get("name", "model")
        # self.set_model_base_properties(config, log_dir) # TODO

        # Logs
        os.makedirs(log_dir, exist_ok=True)
        self.tensorboard = TensorBoard(log_dir)
        self.metric_logs = {"best_epoch": 0, "best_global_step": 0, "best": None, "best_score": 0}
        self.training_logs = {"early_stopping_count": 0}

        # optimization options
        self.grad_max_norm = grad_max_norm

        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = 1
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.learning_rate_scheduler = learning_rate_scheduler
        self.exponential_moving_average = exponential_moving_average
        if exponential_moving_average:
            self.exponential_moving_average = EMA(model, self.exponential_moving_average)

        # property
        self.num_epochs = num_epochs
        self.early_stopping = False
        self.early_stopping_threshold = early_stopping_threshold
        self.max_eval_examples = max_eval_examples
        self.metric_key = metric_key
        self.verbose_step_count = verbose_step_count
        self.eval_and_save_step_count = eval_and_save_step_count
        self.log_dir = log_dir

    def set_model_base_properties(self, config, log_dir):
        model = self.model
        if self.use_multi_gpu:
            model = self.model.module

        model.config = config
        model.log_dir = log_dir
        model.train_counter = self.train_counter
        assert model.is_ready() == True

    def train(self, data_loader, optimizer):
        """ Train """
        # TODO
        print('')
