
from .base import Factory


class DataLoaderFactory(Factory):
    """
    DataLoader Factory Class

    * Args:
        config: data_loader config from argument (config.data_loader)
    """

    def __init__(self, config):
        self.train_batch_size = config.train_batch_size
        self.synthesize_batch_size = config.synthesize_batch_size
        self.cuda_device_id = None
        if config.cuda_devices:
            self.cuda_device_id = config.cuda_devices[0]
