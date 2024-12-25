# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
import os
from typing import Union

from torch.utils.data import ConcatDataset

from .demo_sequence import DemoSequence
from .mot_wrapper import MOT17Wrapper, MOT20Wrapper, MOTS20Wrapper
from .spine_sequence import SpineSequence, SequenceHelper

DATASETS = {}

data_folder = os.getenv('DATASET') if os.getenv('DATASET') else 'spine' 
partition = os.getenv('PARTITION') if os.getenv('PARTITION') else 'test'

# Fill all available datasets, change here to modify / add new datasets.
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for dets in ['DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'MOT17-{split}'
        if dets:
            name = f"{name}-{dets}"
        DATASETS[name] = (
            lambda kwargs, split=split, dets=dets: MOT17Wrapper(split, dets, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08']:
    name = f'MOT20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOT20Wrapper(split, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '05', '06', '07', '09', '11', '12']:
    name = f'MOTS20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOTS20Wrapper(split, **kwargs))

custom_sequences_train = SequenceHelper.get_sequence_names(os.path.join("data", data_folder, "annotations", f"train.json"))
custom_sequences_val = SequenceHelper.get_sequence_names(os.path.join("data", data_folder, "annotations", f"val.json"))
custom_sequences_split = SequenceHelper.get_sequence_names(os.path.join("data", data_folder, "annotations", f"{partition}.json"))

for name in custom_sequences_train:
    DATASETS[name] = (lambda kwargs: [SpineSequence(seq_name=name, 
                                                    partition='train',
                                                    subdir=data_folder,
                                                    **kwargs), ])

for name in custom_sequences_val:
    DATASETS[name] = (lambda kwargs: [SpineSequence(seq_name=name, 
                                                    partition='val',
                                                    subdir=data_folder,
                                                    **kwargs), ])

for name in custom_sequences_split:
    DATASETS[name] = (lambda kwargs: [SpineSequence(seq_name=name, 
                                                    partition=partition,
                                                    subdir=data_folder,
                                                    **kwargs), ])

DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])


class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """Initialize the corresponding dataloader.

        Keyword arguments:
        datasets --  the name of the dataset or list of dataset names
        kwargs -- arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]
            print(f"Single sequence given: {datasets}")

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = [SpineSequence(
                                subdir=data_folder,
                                seq_name=dataset, 
                                partition=partition, 
                                **kwargs), ]
            else:
                self._data = ConcatDataset([self._data, [SpineSequence(
                                                subdir=data_folder,
                                                seq_name=dataset, 
                                                partition=partition, 
                                                **kwargs), ]])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
