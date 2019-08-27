
from NLP.config.registry import Registry

from .base import Factory


class ModelFactory(Factory):
    """
    Model Factory Class

    Create Concrete model according to config.model_name
    Get model from model registries (eg. @register("model:{model_name}"))

    * Args:
        config: model config from argument (config.model)
    """

    def __init__(self, config):
        self.registry = Registry()
        # TODO
        self.name = config.model_name
        self.model_config = {}
        if getattr(config, config.model_name, None):
            self.model_config = vars(getattr(config, config.model_name))

