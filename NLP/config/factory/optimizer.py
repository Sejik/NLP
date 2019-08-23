
from overrides import overrides
import torch

from NLP.config.namespace import NestedNamespace
from NLP.learn.optimization.learning_rate_scheduler import get_lr_schedulers
from NLP.learn.optimization.learning_rate_scheduler import (
    LearningRateWithoutMetricsWrapper,
    LearningRateWithMetricsWrapper,
)
from NLP.learn.optimization.optimizer import get_optimizer_by_name

from .base import Factory


class OptimizerFactory(Factory):
    """
    Optimizer Factory Class

    include optimizer, learning_rate_scheduler and exponential_moving_average

    * Args:
        config: optimizer config from argument (config.optimizer)
    """

    def __init__(self, config):
        # Optimizer
        self.op_type = config.op_type
        self.optimizer_params = {"lr": config.learning_rate}

        op_config = getattr(config, self.op_type, None)
        if op_config is not None:
            op_config = vars(op_config)
            self.optimizer_params.update(op_config)

        # LearningRate Scheduler
        self.lr_schedulers = get_lr_schedulers()
        self.lr_scheduler_type = getattr(config, "lr_scheduler_type", None)
        if self.lr_scheduler_type is not None:
            self.lr_scheduler_config = getattr(config, self.lr_scheduler_type, {})
            if type(self.lr_scheduler_config) == NestedNamespace:
                self.lr_scheduler_config = vars(self.lr_scheduler_config)

        # EMA
        self.ema = getattr(config, "exponential_moving_average", 0)

    @overrides
    def create(self, model):
        if not issubclass(type(model), torch.nn.Module):
            raise ValueError("optimizer model is must be subclass of torch.nn.Module.")

        model_parameters = [param for param in model.parameters() if param.requires_grad]

        optimizer = get_optimizer_by_name(self.op_type)(model_parameters, **self.optimizer_params)
        op_dict = {"optimizer": optimizer}

        # learning_rate_scheduler
        if self.lr_scheduler_type:
            self.lr_scheduler_config["optimizer"] = op_dict["optimizer"]
            lr_scheduler = self.lr_schedulers[self.lr_scheduler_type](**self.lr_scheduler_config)

            if self.lr_scheduler_type == "reduce_on_plateau":
                lr_scheduler = LearningRateWithMetricsWrapper(lr_scheduler)
            else:
                lr_scheduler = LearningRateWithoutMetricsWrapper(lr_scheduler)

            op_dict["learning_rate_scheduler"] = lr_scheduler

        # exponential_moving_average
        if self.ema and self.ema > 0:
            op_dict["exponential_moving_average"] = self.ema

        return op_dict