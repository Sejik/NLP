import argparse
from argparse import RawTextHelpFormatter
import sys

import torch

from NLP.config.namespace import NestedNamespace
from NLP.learn.mode import Mode


def config(argv=None, mode=None):
    if argv is None:
        argv = sys.argv[1:0]

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    general(parser)

    config = parser.parse_args(argv, namespace=NestedNamespace())
    set_gpu_env(config)
    set_batch_size(config)

    return config


def general(parser):
    group = parser.add_argument_group("general")
    group.add_argument(
        "--input_dir",
        type=str, default="data/LJSpeech-1.1", dest="general.input_dir",
        help=""" Raw input directory 
        metadata.csv: metadata(Text explanation) of wav files
        wavs folder: wav files
        """,
    )
    group.add_argument(
        "--dataset",
        type=str, default="ljspeech", dest="general.dataset",
        help=""" Dataset Name [ljspeech] """,
    )
    group.add_argument(
        "--log_dir",
        type=str, default="logs/experiment_1", dest="general.log_dir",
        help=""" TensorBoard and Checkpoint log directory""",
    )

    group.add_argument(
        "--model_name",
        type=str, default="clarinet", dest="general.model_name",
        help=""" model name """,
    )
    group.add_argument(
        "--teacher_model_name",
        type=str, default="wavenet_gaussian", dest="general.teacher_model_name",
        help=""" teacher model name """,
    )
    group.add_argument(
        "--student_model_name",
        type=str, default="wavenet", dest="general.student_model_name",
        help=""" teacher model name """,
    )

    group.add_argument(
        "--load_step",
        type=int, default=0, dest="general.load_step",
        help=""" model load step """,
    )
    group.add_argument(
        "--num_epochs",
        type=int, default=1000, dest="general.num_epochs",
        help=""" number of epochs to train """,
    )
    group.add_argument(
        "--learning_rate",
        type=float, default=0.001, dest="general.learning_rate",
        help=""" the learning rate """,
    )
    group.add_argument(
        "--ema_decay",
        type=float, default=0.9999, dest="general.ema_decay",
        help=""" exponential moving average decay """,
    )

    group.add_argument(
        "--num_blocks",
        type=int, default=2, dest="general.num_blocks",
        help=""" number of blocks """,
    )
    group.add_argument(
        "--num_layers",
        type=int, default=10, dest="general.num_layers",
        help=""" number of layers """,
    )

    group.add_argument(
        "--residual_channels",
        type=int, default=128, dest="general.residual_channels",
        help=""" residual channels """,
    )
    group.add_argument(
        "--gate_channels",
        type=int, default=256, dest="general.gate_channels",
        help=""" gate channels """,
    )
    group.add_argument(
        "--skip_channels",
        type=int, default=128, dest="general.skip_channels",
        help=""" skip channels """,
    )
    group.add_argument(
        "--kernel_size",
        type=int, default=2, dest="general.kernel_size",
        help=""" kernel size """,
    )
    group.add_argument(
        "--cin_channels",
        type=int, default=80, dest="general.cin_channels",
        help=""" cin channels """,
    )

    group.add_argument(
        "--KL_type",
        type=str, default="qp", dest="general.KL_type",
        help=""" pq or qp """,
    )
    group.add_argument(
        "--syn_index",
        type=int, default=0, dest="general.syn_index",
        help=""" index """,
    )
    group.add_argument(
        "--temp",
        type=float, default=1, dest="general.temp",
        help=""" temperature """,
    )

    group.add_argument(
        "--teacher_num_samples",
        type=int, default=5, dest="general.teacher_num_samples",
        help=""" teacher number of samples """,
    )
    group.add_argument(
        "--student_num_samples",
        type=int, default=10, dest="general.student_num_samples",
        help=""" student number of samples """,
    )


def set_gpu_env(config):
    config.use_gpu = torch.cuda.is_available()
    config.gpu_num = len(getattr(config, "cuda_devices", []))

    if not config.use_gpu:
        config.gpu_num = 0
        config.cuda_devices = None


def set_batch_size(config):
    # dynamic batch_size (multi-gpu)
    config.train_batch_size = 8
    config.synthesize_batch_size = 1
    train_batch_size = config.train_batch_size
    synthesize_batch_size = config.synthesize_batch_size
    if config.gpu_num > 1:
        train_batch_size *= config.gpu_num
        synthesize_batch_size *= config.gpu_num
    config.train_batch_size = int(train_batch_size)
    config.synthesize_batch_size = int(synthesize_batch_size)
