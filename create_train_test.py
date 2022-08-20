import os
import subprocess
import urllib

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, fpath)
        if os.stat(fpath).st_size == 0:
            print(f'\n{red}Removing empty file {filename}{white}')
            os.remove(fpath)
    except (urllib.error.URLError, IOError) as e:
        print(f'\n{red}Error downloading {url} {e}{white}')
        os.remove(fpath)


def split_clusters(clusters, feature, test_proportion):
    # Modify the id taking only the uniprot id
    clusters.id = clusters.id.apply(lambda x: x.split('|')[1])

    # Get the sequences that are too long
    too_long = clusters[clusters.length > 4000].id.to_list()

    # Filter clusters and disorder for the sequences that are too long
    clusters = clusters[clusters.length < 4000]
    feature = feature.drop(feature[feature.acc.isin(too_long)].index)

    # Remove clusters that are not in the feature file
    clusters = clusters.drop(clusters[~clusters.id.isin(feature.acc)].index)

    train_ids, test_ids = set(), set()
    for cluster_id in clusters.clstr.unique():
        selected = set(clusters[clusters.clstr == cluster_id].id.to_list())
        if len(test_ids) / (len(train_ids) + len(test_ids) + 1) <= test_proportion:
            test_ids.update(selected)
        else:
            train_ids.update(selected)

    uniprot_to_disprot = {x: y for x, y in zip(feature.acc, feature.disprot_id)}

    return train_ids, test_ids, uniprot_to_disprot


def write_train_test_split(train_ids, test_ids, uniprot_to_disprot, train_file, test_file):
    os.makedirs('data/fastas', exist_ok=True)

    with open(train_file, 'w') as f:
        for uniprot in train_ids:
            f.write(f'{uniprot} {uniprot_to_disprot[uniprot]}\n')

    with open(test_file, 'w') as f:
        for uniprot in test_ids:
            f.write(f'{uniprot} {uniprot_to_disprot[uniprot]}\n')


def split_multifasta(multifasta_path, uniprot_ids):
    multifasta = SeqIO.parse(multifasta_path, 'fasta')
    for record in multifasta:
        id = record.id.split('|')[1]
        if id in uniprot_ids:
            SeqIO.write(record, f'data/fastas/{id}.fasta', 'fasta')


def tr_ts_proportion(tr, ts):
    return len(tr) / (len(tr) + len(ts))


def download_sequences(uniprot_ids):
    os.makedirs('data/fastas', exist_ok=True)
    for uniprot in tqdm(uniprot_ids, desc='Downloading sequences'):
        if not os.path.exists(f'data/fastas/{uniprot}.fasta'):
            download_url(f'https://rest.uniprot.org/uniprotkb/{uniprot}.fasta', 'data/fastas', f'{uniprot}.fasta')


def merge_sequences(uniprot_ids, out_file):
    with open(out_file, 'w') as f:
        for uniprot in uniprot_ids:
            if os.path.exists(f'data/fastas/{uniprot}.fasta'):
                SeqIO.write(SeqIO.parse(f'data/fastas/{uniprot}.fasta', 'fasta'), f, 'fasta')


def remove_too_long_sequences(uniprot_ids):
    for uniprot in uniprot_ids:
        if os.path.exists(f'data/fastas/{uniprot}.fasta'):
            seq = next(SeqIO.parse(f'data/fastas/{uniprot}.fasta', 'fasta'))
            if len(seq.seq) > 4000:
                print(f'{orange}{uniprot} is too long{white}')
                os.remove(f'data/fastas/{uniprot}.fasta')


def create_clusters(feature, feature_name='disorder'):
    uniprot_ids = feature.acc.unique()
    download_sequences(uniprot_ids)
    remove_too_long_sequences(uniprot_ids)

    print(f"Number of final sequences for {feature_name}:", len(os.listdir('data/fastas')))

    merge_sequences(uniprot_ids, out_file=f'data/{feature_name}.fasta')

    os.chdir('data')
    subprocess.call(["../scripts/cd-hit_30.sh", f'{feature_name}_seq.fasta'], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    os.chdir('..')


if __name__ == '__main__':
    green = '\033[92m'
    red = '\033[91m'
    white = '\033[0m'
    orange = '\033[93m'

    # Define train and test set for the disorder feature
    print(f'{green}Working on the disorder feature {white}')
    disorder = pd.read_csv("data/disorder.tsv", sep='\t')
    create_clusters(disorder, feature_name='disorder')

    disorder_clusters = pd.read_csv("data/clusters/disorder_seq-30.clstr", sep='\t')

    train_ids, test_ids, uniprot_to_disprot = split_clusters(disorder_clusters, disorder, test_proportion=0.3)

    p = tr_ts_proportion(train_ids, test_ids)
    print("Disorder:", "Train", len(train_ids), "Test", len(test_ids),
          "Proportion {:.3f} {:.3f}".format(p, 1 - p))

    write_train_test_split(train_ids, test_ids, uniprot_to_disprot,
                           train_file="data/splits/disorder_train.tsv",
                           test_file="data/splits/disorder_test.tsv")

    # Define train and test set for the linker feature
    print(f'\n{green}Working on the linker feature {white}')
    linker = pd.read_csv("data/linker.tsv", sep='\t')
    create_clusters(linker, feature_name='linker')

    linker_clusters = pd.read_csv("data/clusters/linker_seq-30.clstr", sep='\t')

    train_ids, test_ids, uniprot_to_disprot = split_clusters(linker_clusters, linker, test_proportion=0.3)

    p = tr_ts_proportion(train_ids, test_ids)
    print("Linkers:", "Train", len(train_ids), "Test", len(test_ids),
          "Proportion {:.3f} {:.3f}".format(p, 1 - p))

    write_train_test_split(train_ids, test_ids, uniprot_to_disprot,
                           train_file="data/splits/linker_train.tsv",
                           test_file="data/splits/linker_test.tsv")
