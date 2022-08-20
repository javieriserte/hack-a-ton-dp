import pandas as pd
from Bio import SeqIO


def get_sequences(fasta_path):
    sequences = SeqIO.parse(fasta_path, 'fasta')
    return {x.id.split("|")[1]: x.seq for x in sequences}


def create_references(feature, feature_seqs):
    ref_of_id = dict()
    for acc in feature.acc.unique():
        if acc in feature_seqs:
            ref = [0 for _ in range(len(feature_seqs[acc]))]
            for _, row in feature[feature.acc == acc].iterrows():
                for i in range(row.start - 1, row.end):
                    if i < len(ref):
                        ref[i] = 1
            ref_of_id.setdefault(acc, ref)

    return ref_of_id


def write_ref(ref_of_id, sequences, ref_path):
    with open(ref_path, 'w') as f:
        for acc, ref in ref_of_id.items():
            f.write(f'>{acc}\n{sequences[acc]}\n')
            f.write(''.join(map(str, ref)) + '\n')


def create_references_json():
    for feature in ["disorder", "linker"]:
        tr, ts = dict(), dict()
        tr_set = set(pd.read_csv(f"data/splits/{feature}_train.tsv", sep=' ', header=None)[0])
        ts_set = set(pd.read_csv(f"data/splits/{feature}_test.tsv", sep=' ', header=None)[0])

        with open(f"../data/references/{feature}_ref.fasta", "r") as f:
            for line in f:
                if line.startswith(">"):
                    ID = line.strip('>').strip()
                    seq = f.readline().strip()
                    ref = f.readline().strip()
                    if ID in tr_set:
                        tr[ID] = (seq, ref)
                    if ID in ts_set:
                        ts[ID] = (seq, ref)

        tr = pd.DataFrame.from_dict(tr, orient='index', columns=["sequence", "reference"]).reset_index() \
            .rename({'index': 'id'}, axis='columns')
        tr.to_json(f'data/dataset/{feature}_train.json', orient='records')

        ts = pd.DataFrame.from_dict(ts, orient='index', columns=["sequence", "reference"]).reset_index() \
            .rename({'index': 'id'}, axis='columns')
        ts.to_json(f'data/dataset/{feature}_test.json', orient='records')


if __name__ == '__main__':
    # Create disorder references
    disorder = pd.read_csv("data/disorder.tsv", sep='\t')
    disorder_seqs = get_sequences("data/disorder_seq.fasta")

    ref_disorder = create_references(disorder, disorder_seqs)
    print("Number of disorder references:", len(ref_disorder))
    write_ref(ref_disorder, disorder_seqs, "data/references/disorder_ref.fasta")

    # Create linker references
    linker = pd.read_csv("data/linker.tsv", sep='\t')
    linker_seqs = get_sequences("data/linker_seq.fasta")

    ref_linker = create_references(linker, linker_seqs)
    print("Number of linker references:", len(ref_linker))
    write_ref(ref_linker, linker_seqs, "data/references/linker_ref.fasta")

    create_references_json()
