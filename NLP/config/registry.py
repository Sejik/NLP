
from NLP.config.pattern import Singleton


class Registry(metaclass=Singleton):
    """
    Registry class (Singleton)
    """

    def __init__(self):
        self._name_to_subclass = {
            "reader": {},
        }

    def add(self, name, obj):
        component_type, component_name = self._split_component_type_and_name(name)
        self._name_to_subclass[component_type][component_name] = obj

    def get(self, name):
        component_type, component_name = self._split_component_type_and_name(name)

        if component_type not in self._name_to_subclass:
            raise ValueError(f"There is no {component_type} in _name_to_subclass.")
        if component_name not in self._name_to_subclass[component_type]:
            raise ValueError(f"There is no {component_name} object in {component_type}.")
        return self._name_to_subclass[component_type][component_name]

    def _split_component_type_and_name(self, name):
        if ":" in name:
            names = name.split(":")
            return names[0], names[1]
        else:
            raise ValueError("do not recognize component_type.")

