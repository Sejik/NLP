
from NLP.data.dataset.base import DatasetBase


class LJSpeechDataset(DatasetBase):
    """
    LJSpeech DataReader

    * Args:
        file_path:
    """

    def __init__(self, batch, helper=None):
        super(LJSpeechDataset, self).__init()

        self.name = "ljspeech"
        self.helper = helper
        self.raw_dataset = helper["raw_dataset"]

        # Features

        # Labels
