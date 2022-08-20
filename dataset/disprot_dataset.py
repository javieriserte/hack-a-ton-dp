import os
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.encoding import Uniprot21
from dataset.utils import parse_target, read_pssm


class Sequence:
    def __init__(self, seq_id, sequence, target=None, load_pssm=False, pssm_path=None, data_transform=None,
                 target_transform=None):
        self.seq_id = seq_id
        self.sequence = sequence
        self.encoded_sequence = torch.tensor(Uniprot21().encode(sequence), dtype=torch.uint8).reshape(1, -1)
        if target is not None:
            self._target = parse_target(target)

        self.data_transform = data_transform
        self.target_transform = target_transform

        self.pssm = None
        if load_pssm and pssm_path is not None:
            self.pssm = read_pssm(os.path.join(pssm_path, 'pssm/{}.pssm'.format(self.seq_id)))

    @property
    def data(self):
        if self.pssm is not None:
            data = torch.cat((self.encoded_sequence, self.pssm.T), dim=0)
        else:
            data = self.encoded_sequence

        if self.data_transform is not None:
            data = self.data_transform(data)
        return data.float()

    @property
    def target(self):
        target = self._target
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target.float()

    @property
    def clean_target(self):
        return self._target.numpy()

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        return 'Sequence({}, {})'.format(self.seq_id, self.sequence)

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, i):
        return self.data, self.target

    def as_dict(self):
        return {"seq_id": self.seq_id, "sequence": self.sequence, "target": self.target, "data": self.data}


# Base class for the two datasets, with common functionality
class DisprotDataset(Dataset):
    def __init__(self, dataset_root='../data/dataset', feature_root='../data/features', train_file=None, test_file=None,
                 train=True, transform=None, target_transform=None, pssm=False):
        # Define the encoder fot the sequence
        self.encoder = Uniprot21()

        self.train = train

        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.raw_data = pd.read_json(os.path.join(dataset_root, train_file), orient='records', dtype=False)
        else:
            self.raw_data = pd.read_json(os.path.join(dataset_root, test_file), orient='records', dtype=False)

        # Create sequences objects
        self.data = []
        split = 'train' if self.train else 'test'
        for seq_id, sequence, target in tqdm(self.raw_data.itertuples(index=False), desc=f'Importing {split} sequences',
                                             total=len(self.raw_data)):
            # Split the sequence with a moving window of size 20, add padding at the end
            self.data.append(Sequence(seq_id, sequence, target, load_pssm=pssm, pssm_path=feature_root,
                                      data_transform=self.transform, target_transform=self.target_transform))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DisorderDataset(DisprotDataset):
    def __init__(self, dataset_root='../data/dataset', feature_root='../data/features', train=True, transform=None,
                 target_transform=None, pssm=False):
        super(DisorderDataset, self).__init__(dataset_root=dataset_root, feature_root=feature_root,
                                              train=train, transform=transform,
                                              target_transform=target_transform, pssm=pssm,
                                              train_file='disorder_train.json',
                                              test_file='disorder_test.json', )


class LinkerDataset(DisprotDataset):
    def __init__(self, dataset_root='../data/dataset', feature_root='../data/features', train=True, transform=None,
                 target_transform=None, pssm=False):
        super(LinkerDataset, self).__init__(dataset_root=dataset_root, feature_root=feature_root,
                                            train=train, transform=transform,
                                            target_transform=target_transform, pssm=pssm,
                                            train_file='linker_train.json',
                                            test_file='linker_test.json', )


def collate_fn(batch: List[Sequence]):
    data = torch.stack([item.data for item in batch])
    target = torch.stack([item.target for item in batch])
    return batch, data, target


if __name__ == '__main__':
    # Disorder Dataset
    train_disorder = DisorderDataset(train=True, pssm=False)
    test_disorder = DisorderDataset(train=False, pssm=False)

    disorder_dataloader = DataLoader(test_disorder, batch_size=1, shuffle=False, num_workers=0)

    for i_batch, (X, y) in enumerate(disorder_dataloader):
        print("Features:", X, "Target:", y)
