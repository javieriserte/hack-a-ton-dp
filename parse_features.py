import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_pssm(pssm_path):
    pssm = pd.read_csv(pssm_path, sep='\s+', skiprows=3, header=None)
    pssm = pssm.dropna().values[:, 2:22].astype(np.int8)

    return pssm


def read_hhm(hhm_path):
    # Find where the hhm matrix starts
    count = 3
    with open(hhm_path, 'r') as f:
        for line in f:
            if line[:3] == 'HMM':
                break
            else:
                count += 1

    # Read the hhm matrix, skipping the last line
    hhm = pd.read_csv(hhm_path, sep='\s+', header=None, skiprows=count)
    # Bring the second sub-line to the first line, remove the remainder of the second sub-line
    hhm = hhm.values[:-1, :22].reshape(-1, 44)[:, 2:-12]
    # Remove * from the matrix and convert to float
    hhm[hhm == '*'] = 9999
    hhm = hhm.astype(np.int16)

    return hhm


def spd3_feature_sincos(x, seq):
    ASA = x[:, 0]
    asa_rnm = {'A': 115, 'C': 135, 'D': 150, 'E': 190, 'F': 210, 'G': 75, 'H': 195, 'I': 175, 'K': 200, 'L': 170,
               'M': 185, 'N': 160, 'P': 145, 'Q': 180, 'R': 225, 'S': 115, 'T': 140, 'V': 155, 'W': 255, 'Y': 230}

    ASA_div = np.array([asa_rnm[aa] if aa in asa_rnm else 1 for aa in seq])
    ASA = (ASA / ASA_div)[:, None]
    angles = x[:, 1:5]
    HCEprob = x[:, -3:]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles), np.cos(angles)], 1)
    return np.concatenate([ASA, angles, HCEprob], 1)


def read_spd3(spd3_path):
    spd3_features = pd.read_csv(spd3_path, sep='\s+')
    seq = spd3_features.AA.to_list()
    spd3_features = spd3_features.values[:, 3:].astype(float)
    spd3 = spd3_feature_sincos(spd3_features, seq)
    return spd3


def read_phy7(sequence):
    phy7 = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
            'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
            'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
            'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
            'F': [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
            'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
            'H': [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
            'I': [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
            'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
            'L': [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
            'M': [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
            'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
            'P': [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
            'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
            'R': [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
            'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
            'T': [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
            'V': [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
            'W': [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
            'Y': [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}

    return np.array([phy7[aa] if aa in phy7 else [0, 0, 0, 0, 0, 0, 0] for aa in sequence])


if __name__ == '__main__':
    spd3_files = list(Path('data/features/spd3').glob('*.spd3'))
    for spd3_file in tqdm(spd3_files, total=len(spd3_files)):
        try:
            sp3 = read_spd3(spd3_file)
            break
        except Exception as e:
            print(f'{spd3_file} failed: {e}')
            continue

    pssm_files = list(Path('data/features/pssm').glob('*.pssm'))
    for pssm_file in tqdm(pssm_files, total=len(pssm_files)):
        try:
            pssm = read_pssm(pssm_file)
            break
        except Exception as e:
            print(f'{pssm_file} failed: {e}')
            continue

    hhm_files = list(Path('data/features/hhm').glob('*.hhm'))
    for hhm_file in tqdm(hhm_files, total=len(hhm_files)):
        try:
            hhm = read_hhm(hhm_file)
            break
        except Exception as e:
            print(f'{hhm_file} failed: {e}')
            continue
