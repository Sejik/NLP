
from torch.utils.data.dataset import Dataset


class DatasetBase(Dataset):
    """
    Dataset Base Model
    An abstract class representing a Dataset.
    """

    def __init__(self):
        # Features - Lazy Evalutation
        self.f_count = 0
        self.features = []
