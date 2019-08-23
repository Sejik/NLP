
import argparse
from argparse import RawTextHelpFormatter
import sys

import torch

from NLP.config import utils
from NLP.learn.mode import Mode
from NLP.config.namespace import NestedNamespace


def config(argv=None, mode=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    general(parser)

    if mode == Mode.EVAL:
        evaluate(parser)
        return parser.parse_args(argv, namespace=NestedNamespace())

    return train_config(parser, input_argv=argv)


def train_config(parser, input_argv=None):
    data(parser)
    model(parser)

    trainer(parser)

    config = parser.parse_args(input_argv, namespace=NestedNamespace())

    # overwrite input argument when base_config and arguments are provided.
    # (eg. --base_config bidaf --learning_rate 2) -> set bidaf.json then overwrite learning_rate 2)
    input_args = get_input_arguments(parser, input_argv)
    for k, v in input_args.items():
        setattr(config, k, v)

    set_gpu_env(config)
    set_batch_size(config)
    return config


def get_input_arguments(parser, input_arguments):
    flat_config = parser.parse_args(input_arguments)
    config_dict = utils.convert_config2dict(flat_config)
    config_default_none = {k: None for k in config_dict.keys()}

    input_parser = argparse.ArgumentParser(parents=[parser], conflict_handler="resolve")
    input_parser.set_defaults(**config_default_none)

    input_config = input_parser.parse_args(input_arguments)
    input_config = utils.convert_config2dict(input_config)

    if "base_config" in input_config:
        del input_config["base_config"]
    return {k: v for k, v in input_config.items() if v is not None}


def set_gpu_env(config):
    # GPU
    config.use_gpu = torch.cuda.is_available()

    config.gpu_num = len(getattr(config, "cuda_devices", []))

    if not config.use_gpu:
        config.gpu_num = 0
        config.cuda_devices = None


def set_batch_size(config):
    # dynamic batch_size (multi-gpu and gradient_accumulation_steps)
    batch_size = config.iterator.batch_size
    if config.gpu_num > 1:
        batch_size *= config.gpu_num
    if getattr(config.optimizer, "gradient_accumulation_steps", None):
        batch_size = batch_size // config.optimizer.gradient_accumulation_steps
    config.iterator.batch_size = int(batch_size)


def arg_str2bool(v):
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def general(parser):

    group = parser.add_argument_group("General")
    group.add_argument(
        "--seed_num",
        type=int, default=21, dest="seed_num",
        help=""" Manually set seed_num (Python, Numpy, Pytorch) default is 21 """,
    )
    group.add_argument(
        "--cuda_devices", nargs="+",
        type=int, default=[], dest="cuda_devices",
        help=""" Set cuda_devices ids (use GPU). if you use NSML, use GPU_NUM""",
    )
    group.add_argument(
        "--slack_url",
        type=str, default=None, dest="slack_url",
        help=""" Slack notification (Incoming Webhook) """,
    )


def evaluate(parser):

    group = parser.add_argument_group("Run evaluate")
    group.add_argument(
        "data_file_path",
        type=str,
        help=" Path to the file containing the evaluation data"
    )
    group.add_argument("checkpoint_path", type=str, help="Path to an checkpoint trained model")
    group.add_argument(
        "--infer",
        default=None, dest="inference_latency", type=int,
        help=""" Evaluate with inference-latency with maximum value (ms)""",
    )
    group.add_argument(
        "--prev_cuda_device_id",
        type=int, default=0, dest="prev_cuda_device_id",
        help=""" Previous cuda device id (need to mapping)""",
    )


def data(parser):
    group = parser.add_argument_group("Data Reader")
    group.add_argument(
        "--dataset",
        type=str, default="ljspeech", dest="data_reader.dataset",
        help=""" Dataset Name [ljspeech] """,
    )
    group.add_argument(
        "--train_file_path",
        type=str, default="LJSpeech-1.1/", dest="data_reader.train_file_path",
        help=""" train file path. """,
    )

    # group = parser.add_argument_group("  # LJSpeech DataSet")

    group = parser.add_argument_group("Iterator")
    group.add_argument(
        "--batch_size", '-b', type=int, default=8, dest="iterator.batch_size",
        help=""" Maximum batch size for trainer""",
    )


def model(parser):

    group = parser.add_argument_group("Model")
    group.add_argument(
        "--model_name",
        type=str, default="clarinet", dest="model.name",
        help="""\

    Pre-defined model

    * Text-to-Speech
      [clarient]
    """,
    )

    tts_title = "ㅁText to Speech"
    group = parser.add_argument_group(f"{tts_title}\n # ClariNet")
    group.add_argument(
        "--clarinet.num_block",
        type=int, default=2, dest="model.clarinet.num_block",
        help="""The number of ClariNet model blocks"""
    )
    group.add_argument(
        "--clarinet.num_layers",
        type=int, default=10, dest="model.clarinet.num_layers",
        help="""The number of ClariNet layers"""
    )
    group.add_argument(
        "--clarinet.residual_channels",
        type=int, default=128, dest="model.clarinet.residual_channels",
        help="The number of ClariNet residual channels"
    )
    group.add_argument(
        "--clarinet.gate_channels",
        type=int, default=256, dest="model.clarinet.gate_channels",
        help="""The number of ClariNet model gate channels"""
    )
    group.add_argument(
        "--clarinet.skip_channels",
        type=int, default=128, dest="model.clarinet.skip_channels",
        help="""The number of ClariNet layers"""
    )
    group.add_argument(
        "--clarinet.cin_channels",
        type=int, default=2, dest="model.clarinet.cin_channels",
        help="The size of ClariNet base"
    )


def trainer(parser):

    group = parser.add_argument_group("Trainer")
    group.add_argument(
        "--num_epochs", "-e",
        type=int, default=1000, dest="trainer.num_epochs",
        help=""" The number of training epochs""",
    )
    group.add_argument(
        "--save_epoch_count",
        type=int, default=1, dest="trainer.save_epoch_count",
        help=""" The number of save epoch count""",
    )
    group.add_argument(
        "--log_dir",
        type=str, default="logs/experiment_1", dest="trainer.log_dir",
        help=""" TensorBoard and Checkpoint log directory""",
    )

    group = parser.add_argument_group("Optimizer")
    group.add_argument(
        "--optimizer_type",
        type=str, default="adam", dest="optimizer.op_type",
        help=""" Optimizer
        (https://pytorch.org/docs/stable/optim.html#algorithms)

        - adadelta: ADADELTA: An Adaptive Learning Rate Method
            (https://arxiv.org/abs/1212.5701)
        - adagrad: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
            (http://jmlr.org/papers/v12/duchi11a.html)
        - adam: Adam: A Method for Stochastic Optimization
            (https://arxiv.org/abs/1412.6980)
        - sparse_adam: Implements lazy version of Adam algorithm suitable for sparse tensors.
            In this variant, only moments that show up in the gradient get updated,
            and only those portions of the gradient get applied to the parameters.
        - adamax: Implements Adamax algorithm (a variant of Adam based on infinity norm).
        - averaged_sgd: Acceleration of stochastic approximation by averaging
            (http://dl.acm.org/citation.cfm?id=131098)
        - rmsprop: Implements RMSprop algorithm.
            (https://arxiv.org/pdf/1308.0850v5.pdf)
        - rprop: Implements the resilient backpropagation algorithm.
        - sgd: Implements stochastic gradient descent (optionally with momentum).
            Nesterov momentum: (http://www.cs.toronto.edu/~hinton/absps/momentum.pdf)

        [adadelta|adagrad|adam|sparse_adam|adamax|averaged_sgd|rmsprop|rprop|sgd]""",
    )
    group.add_argument(
        "--learning_rate", '-lr',
        type=float, default=0.001, dest="optimizer.learning_rate",
        help="""\
        Starting learning rate.
        Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001 """,
    )

    group = parser.add_argument_group("  # Adadelta")
    group.add_argument(
        "--adadelta.rho",
        type=float, default=0.9, dest="optimizer.adadelta.rho",
        help="""\
        coefficient used for computing a running average of squared gradients
        Default: 0.9 """,
    )
    group.add_argument(
        "--adadelta.eps",
        type=float, default=1e-6, dest="optimizer.adadelta.eps",
        help="""\
        term added to the denominator to improve numerical stability
        Default: 1e-6 """,
    )
    group.add_argument(
        "--adadelta.weight_decay",
        type=float,
        default=0,
        dest="optimizer.adadelta.weight_decay",
        help="""\
        weight decay (L2 penalty)
        Default: 0 """,
    )

    group = parser.add_argument_group("  # Adagrad")
    group.add_argument(
        "--adagrad.lr_decay",
        type=float, default=0, dest="optimizer.adagrad.lr_decay",
        help="""\
        learning rate decay
        Default: 0 """,
    )
    group.add_argument(
        "--adagrad.weight_decay",
        type=float,
        default=0,
        dest="optimizer.adagrad.weight_decay",
        help="""\
        weight decay (L2 penalty)
        Default: 0 """,
    )

    group = parser.add_argument_group("  # Adam")
    group.add_argument(
        "--adam.betas", nargs="+",
        type=float, default=[0.9, 0.999], dest="optimizer.adam.betas",
        help="""\
        coefficients used for computing running averages of gradient and its square
        Default: (0.9, 0.999) """,
    )
    group.add_argument(
        "--adam.eps",
        type=float, default=1e-8, dest="optimizer.adam.eps",
        help="""\
        term added to the denominator to improve numerical stability
        Default: 1e-8 """,
    )
    group.add_argument(
        "--adam.weight_decay",
        type=float,
        default=0,
        dest="optimizer.adam.weight_decay",
        help="""\
        weight decay (L2 penalty)
        Default: 0 """,
    )

    group = parser.add_argument_group("  # SparseAdam")
    group.add_argument(
        "--sparse_adam.betas", nargs="+",
        type=float, default=[0.9, 0.999], dest="optimizer.sparse_adam.betas",
        help="""\
        coefficients used for computing running averages of gradient and its square
        Default: (0.9, 0.999) """,
    )
    group.add_argument(
        "--sparse_adam.eps",
        type=float, default=1e-8, dest="optimizer.sparse_adam.eps",
        help="""\
        term added to the denominator to improve numerical stability
        Default: 1e-8 """,
    )

    group = parser.add_argument_group("  # Adamax")
    group.add_argument(
        "--adamax.betas", nargs="+",
        type=float, default=[0.9, 0.999], dest="optimizer.adamax.betas",
        help="""\
        coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999) """,
    )
    group.add_argument(
        "--adamax.eps",
        type=float, default=1e-8, dest="optimizer.adamax.eps",
        help="""\
        term added to the denominator to improve numerical stability.
        Default: 1e-8 """,
    )
    group.add_argument(
        "--adamax.weight_decay",
        type=float, default=0, dest="optimizer.adamax.weight_decay",
        help="""\
        weight decay (L2 penalty)
        Default: 0 """,
    )

    group = parser.add_argument_group("  # ASGD (Averaged Stochastic Gradient Descent)")
    group.add_argument(
        "--averaged_sgd.lambd",
        type=float, default=1e-4, dest="optimizer.averaged_sgd.lambd",
        help="""\
        decay term
        Default: 1e-4 """,
    )
    group.add_argument(
        "--averaged_sgd.alpha",
        type=float, default=0.75, dest="optimizer.averaged_sgd.alpha",
        help="""\
        power for eta update
        Default: 0.75 """,
    )
    group.add_argument(
        "--averaged_sgd.t0",
        type=float, default=1e6, dest="optimizer.averaged_sgd.t0",
        help="""\
        point at which to start averaging
        Default: 1e6 """,
    )
    group.add_argument(
        "--averaged_sgd.weight_decay",
        type=float, default=0, dest="optimizer.averaged_sgd.weight_decay",
        help="""\
        weight decay (L2 penalty)
        Default: 0 """,
    )

    group = parser.add_argument_group("  # RMSprop")
    group.add_argument(
        "--rmsprop.momentum",
        type=float, default=0, dest="optimizer.rmsprop.momentum",
        help="""\
        momentum factor
        Default: 0 """,
    )
    group.add_argument(
        "--rmsprop.alpha",
        type=float, default=0.99, dest="optimizer.rmsprop.alpha",
        help="""\
        smoothing constant
        Default: 0.99 """,
    )
    group.add_argument(
        "--rmsprop.eps",
        type=float, default=1e-8, dest="optimizer.rmsprop.eps",
        help="""\
        term added to the denominator to improve numerical stability.
        Default: 1e-8 """,
    )
    group.add_argument(
        "--rmsprop.centered",
        type=arg_str2bool, default=False, dest="optimizer.rmsprop.centered",
        help="""\
        if True, compute the centered RMSProp,
        the gradient is normalized by an estimation of its variance
        Default: False """,
    )
    group.add_argument(
        "--rmsprop.weight_decay",
        type=float, default=0, dest="optimizer.rmsprop.weight_decay",
        help="""\
        weight decay (L2 penalty)
        Default: 0 """,
    )

    group = parser.add_argument_group("  # SGD (Stochastic Gradient Descent)")
    group.add_argument(
        "--sgd.momentum",
        type=float, default=0, dest="optimizer.sgd.momentum",
        help="""\
        momentum factor
        Default: 0 """,
    )
    group.add_argument(
        "--sgd.dampening",
        type=float, default=0, dest="optimizer.sgd.dampening",
        help="""\
        dampening for momentum
        Default: 0 """,
    )
    group.add_argument(
        "--sgd.nesterov",
        type=arg_str2bool, default=False, dest="optimizer.sgd.nesterov",
        help="""\
        enables Nesterov momentum
        Default: False """,
    )
    group.add_argument(
        "--sgd.weight_decay",
        type=float, default=0, dest="optimizer.sgd.weight_decay",
        help="""\
        weight decay (L2 penalty)
        Default: 0 """,
    )

    group = parser.add_argument_group("Learning Rate Scheduler")
    group.add_argument(
        "--lr_scheduler_type",
        type=str, default=None, dest="optimizer.lr_scheduler_type",
        help="""Learning Rate Schedule
        (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) \n

        - lambda: Sets the learning rate of each parameter group to the
            initial lr times a given function.
        - step: Sets the learning rate of each parameter group to the
            initial lr decayed by gamma every step_size epochs.
        - multi_step: Set the learning rate of each parameter group to
            the initial lr decayed by gamma once the number of epoch
            reaches one of the milestones.
        - exponential: Set the learning rate of each parameter group to
            the initial lr decayed by gamma every epoch.
        - cosine: Set the learning rate of each parameter group using
            a cosine annealing schedule, where ηmax is set to the initial
            lr and Tcur is the number of epochs since the last restart in SGDR:
            SGDR: Stochastic Gradient Descent with Warm Restarts
            (https://arxiv.org/abs/1608.03983)
        When last_epoch=-1, sets initial lr as lr.

        - reduce_on_plateau: Reduce learning rate when a metric has
            stopped improving. Models often benefit from reducing the
            learning rate by a factor of 2-10 once learning stagnates.
            This scheduler reads a metrics quantity and if no improvement
            is seen for a ‘patience’ number of epochs, the learning rate is reduced.
        - warmup: a learning rate warm-up scheme with an inverse exponential increase
             from 0.0 to {learning_rate} in the first {final_step}.

        [step|multi_step|exponential|reduce_on_plateau|cosine|warmup]
            """,
    )

    group = parser.add_argument_group("  # StepLR")
    group.add_argument(
        "--step.step_size",
        type=int, default=1, dest="optimizer.step.step_size",
        help="""\
        Period of learning rate decay.
        Default: 1""",
    )
    group.add_argument(
        "--step.gamma",
        type=float, default=0.1, dest="optimizer.step.gamma",
        help="""\
        Multiplicative factor of learning rate decay.
        Default: 0.1. """,
    )
    group.add_argument(
        "--step.last_epoch",
        type=int, default=-1, dest="optimizer.step.last_epoch",
        help="""\
        The index of last epoch.
        Default: -1. """
    )

    group = parser.add_argument_group("  # MultiStepLR")
    group.add_argument(
        "--multi_step.milestones", nargs="+",
        type=int, dest="optimizer.multi_step.milestones",
        help="""\
        List of epoch indices. Must be increasing
        list of int""",
    )
    group.add_argument(
        "--multi_step.gamma",
        type=float, default=0.1, dest="optimizer.multi_step.gamma",
        help="""\
        Multiplicative factor of learning rate decay.
        Default: 0.1. """,
    )
    group.add_argument(
        "--multi_step.last_epoch",
        type=int, default=-1, dest="optimizer.multi_step.last_epoch",
        help="""\
        The index of last epoch.
        Default: -1. """
    )

    group = parser.add_argument_group("  # ExponentialLR")
    group.add_argument(
        "--exponential.gamma",
        type=float, default=0.1, dest="optimizer.exponential.gamma",
        help="""\
        Multiplicative factor of learning rate decay.
        Default: 0.1. """,
    )
    group.add_argument(
        "--exponential.last_epoch",
        type=int, default=-1, dest="optimizer.exponential.last_epoch",
        help="""\
        The index of last epoch.
        Default: -1. """
    )

    group = parser.add_argument_group("  # CosineAnnealingLR")
    group.add_argument(
        "--cosine.T_max",
        type=int, default=50, dest="optimizer.cosine.T_max",
        help="""\
        Maximum number of iterations.
        Default: 50""",
    )
    group.add_argument(
        "--cosine.eta_min",
        type=float, default=0, dest="optimizer.cosine.eta_min",
        help="""\
        Minimum learning rate.
        Default: 0. """,
    )
    group.add_argument(
        "--cosine.last_epoch",
        type=int, default=-1, dest="optimizer.cosine.last_epoch",
        help="""\
        The index of last epoch.
        Default: -1. """
    )

    group = parser.add_argument_group("  # ReduceLROnPlateau")
    group.add_argument(
        "--reduce_on_plateau.factor",
        type=float, default=0.1, dest="optimizer.reduce_on_plateau.factor",
        help=""" Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1. """,
    )
    group.add_argument(
        "--reduce_on_plateau.mode",
        type=str, default="min", dest="optimizer.reduce_on_plateau.mode",
        help="""\
        One of `min`, `max`. In `min` mode, lr will
        be reduced when the quantity monitored has stopped
        decreasing; in `max` mode it will be reduced when the
        quantity monitored has stopped increasing.
        Default: 'min'. """,
    )
    group.add_argument(
        "--reduce_on_plateau.patience",
        type=int, default=10, dest="optimizer.reduce_on_plateau.patience",
        help="""\
        Number of epochs with no improvement after which learning rate will be reduced.
        Default: 10. """,
    )
    group.add_argument(
        "--reduce_on_plateau.threshold",
        type=float, default=1e-4, dest="optimizer.reduce_on_plateau.threshold",
        help="""\
        Threshold for measuring the new optimum, to only focus on significant changes.
        Default: 1e-4 """,
    )
    group.add_argument(
        "--reduce_on_plateau.threshold_mode",
        type=str, default="rel", dest="optimizer.reduce_on_plateau.threshold_mode",
        help="""\
        One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or
        best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold
        in max mode or best - threshold in min mode.
        Default: ‘rel’. """
    )
    group.add_argument(
        "--reduce_on_plateau.cooldown",
        type=int, default=0, dest="optimizer.reduce_on_plateau.cooldown",
        help="""\
        Number of epochs to wait before resuming normal operation after lr has been reduced.
        Default: 0. """,
    )
    group.add_argument(
        "--reduce_on_plateau.min_lr", nargs="+",
        type=float, default=0, dest="optimizer.reduce_on_plateau.min_lr",
        help="""\
        A scalar or a list of scalars. A lower bound on the learning rate of
        all param groups or each group respectively.
        Default: 0. """,
    )
    group.add_argument(
        "--reduce_on_plateau.eps",
        type=float, default=1e-8, dest="optimizer.reduce_on_plateau.eps",
        help="""\
        Minimal decay applied to lr. If the difference between new and
        old lr is smaller than eps, the update is ignored.
        Default: 1e-8 """,
    )

    group = parser.add_argument_group("  # WarmUpLR")
    group.add_argument(
        "--warmup.final_step",
        type=int, default=1000, dest="optimizer.warmup.final_step",
        help="""\
        The number of steps to exponential increase the learning rate.
        Default: 1000. """,
    )
    group.add_argument(
        "--warmup.last_epoch",
        type=int, default=-1, dest="optimizer.warmup.last_epoch",
        help="""\
        The index of last epoch.
        Default: -1. """
    )

    group = parser.add_argument_group("Exponential Moving Average")
    group.add_argument(
        "--ema",
        type=float, default=None, dest="optimizer.exponential_moving_average",
        help="""\
        Exponential Moving Average
        Default: None (don't use)""",
    )
    group.add_argument(
        "--ema_decay", type=float, default=0.9999, dest="optimizer.ema.decay",
        help=""" Exponential Moving Average Decay""",
    )