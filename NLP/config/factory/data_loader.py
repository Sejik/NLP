
from overrides import overrides
from torch.utils.data import DataLoader

from .base import Factory


class DataLoaderFactory(Factory):
    """
    DataLoader Factory Class

    * Args:
        config: data_loader config from argument (config.data_loader)
    """

    def __init__(self, config):
        self.train_batch_size = config.train_batch_size
        self.synthesize_batch_size = config.synthesize_batch_size
        self.cuda_device_id = None
        if config.cuda_devices:
            self.cuda_device_id = config.cuda_devices[0]

    @overrides
    def create(self, datasets):
        """ create train iterator """
        dataset_key = next(iter(datasets))
        dataset = datasets[dataset_key]

        if getattr(dataset, "name", None) is None:
            raise ValueError("unknown dataset.")

        train_loader = None
        if "train" in datasets:
            train_loader, synthesize_loader = self.make_data_loader(
                datasets["train"],
                train_batch_size=self.train_batch_size,
                synthesize_batch_size=self.synthesize_batch_size,
                shuffle=True,
                cuda_device_id=self.cuda_device_id
            )

        return train_loader, synthesize_loader

    def make_data_loader(self, dataset, train_batch_size=8, synthesize_batch_size=1, shuffle=True, cuda_device_id=None):
        is_cpu = cuda_device_id is None

        # TODO: how to give the dataset
        return DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn(cuda_device_id=cuda_device_id),
            num_workers=0,
            pin_memory=is_cpu,  # only CPU memory can be pinned
        ), DataLoader(
            dataset,
            batch_size=synthesize_batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn(cuda_device_id=cuda_device_id),
            num_workers=0,
            pin_memory=is_cpu,  # only CPU memory can be pinned
        )