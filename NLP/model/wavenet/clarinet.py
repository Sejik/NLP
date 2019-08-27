
from overrides import overrides

from NLP.decorator import register
from NLP.model.base import ModelBase  # TODO


@register("model:clarinet")
class ClariNet(ModelBase):
    """
    TTS Model.
    
    * Args:
    
    * Kwargs:
    """

    def __init__(self):
        super(ClariNet, self).__init__()
        # TODO: model define
        # preprocessing
        # modeling

    @overrides
    def forward(self, inputs):
        # TODO: model forward
        print('')

