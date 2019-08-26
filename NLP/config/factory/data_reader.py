
from overrides import overrides

from NLP.config.registry import Registry

from .base import Factory


class DataReaderFactory(Factory):
    """
    DataReader Factory Class

    Create Concrete reader according to config.dataset
    Get reader from reader registries (eg. @register("reader:{reader_name}"))

    * Args:
        config: data_reader config from argument (config.data_reader)
    """

    def __init__(self, config):
        self.registry = Registry()

        self.dataset = config.dataset
        file_paths = {}
        if getattr(config, "input_dir") and config.input_dir != "":
            file_paths["input"] = config.input_dir

        self.reader_config = {"file_paths": file_paths}

    @overrides
    def create(self):
        reader = self.registry.get(f"reader:{self.dataset.lower()}")
        return reader(**self.reader_config)

