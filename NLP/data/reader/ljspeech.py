
from NLP.data.dataset import LJSpeechDataset  # TODO
from NLP.data.reader.base import DataReader  # TODO
from NLP.decorator import register


@register("reader:ljspeech")
class LJSpeechReader(DataReader):
    """
    LJSpeech DataReader

    * Args:
        file_paths: file paths
    """

    def __init__(self, file_paths):  # TODO
        super(LJSpeechReader, self).__init__(file_paths, LJSpeechDataset)

