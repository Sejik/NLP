
from overrides import overrides

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

        self.name = config.name
        self.model_config = {}
        if getattr(config, config.name, None):
            self.model_config = vars(getattr(config, config.name))

        self.is_independent = getattr(config, "independent", False)

    @overrides
    def create(self, token_makers, **params):
        model = self.registry.get(f"model:{self.name}")

        return model(**self.model_config, **params)