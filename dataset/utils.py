import numpy as np
import pandas as pd
import torch


class PadRight(object):
    """Pad the tensor to a given size.

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        padding = self.output_size - sample.size()[-1]
        return torch.nn.functional.pad(sample, (0, padding), 'constant', 0)


def read_pssm(pssm_path):
    pssm = pd.read_csv(pssm_path, sep='\s+', skiprows=3, header=None)
    pssm = pssm.dropna().values[:, 2:22].astype(np.int8)
    pssm = torch.tensor(pssm, dtype=torch.uint8)

    return pssm


def parse_target(x):
    return torch.tensor([int(y) for y in x], dtype=torch.uint8)
