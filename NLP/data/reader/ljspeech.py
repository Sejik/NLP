
import logging

from overrides import overrides

from NLP.data.dataset import LJSpeechDataset
from NLP.data.reader.base import DataReader
from NLP.decorator import register

logger = logging.getLogger(__name__)


@register("reader:ljspeech")
class LJSpeechReader(DataReader):
    """
    LJSpeech DataReader

    * Args:
        file_path: file folder (train)
    """

    def __init__(self, file_paths):
        super(LJSpeechReader, self).__init__(file_paths, LJSpeechDataset)

    @overrides
    def _read(self, file_path, data_type=None):
        data = self.data_handler.read(file_path, return_path=True)
        # data path


        # load data
        ljspeech = 0  # !!!!!!!!
        # preprocessing & (text and mel matching)

        helper = {
            "file_path": file_path,
            "raw_dataset": ljspeech,
        }


