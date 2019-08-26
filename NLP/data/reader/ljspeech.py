
import os

from overrides import overrides

from NLP.data.dataset import LJSpeechDataset  # TODO
from NLP.data.batch import make_batch  # TODO
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

    @overrides
    def _read(self, file_path, data_type=None):
        data = self.data_handler.read(file_path, return_path=True)
        ljspeech = os.path.join(data, "metadata.csv")

        helper = {
            "file_path": file_path,
            "examples": {},
            "metadata_path": ljspeech,
        }

        features, labels = [], []
        with open(ljspeech) as f:
            for line in f:
                parts = line.strip().split('|')
                wav_path = data / 'wavs' / ('%s.wav' % parts[0])  # TODO : check value
                text = parts[2]

                future_row = {
                    "context": text
                }
                features.append(future_row)
                label_row = {
                    "wav_name": wav_path
                    # TODO: maybe npy needed
                }
                labels.append(label_row)

        helper["examples"] = {
            "context": text,
            "wav_name": wav_path
        }

        return make_batch(features, labels), helper


