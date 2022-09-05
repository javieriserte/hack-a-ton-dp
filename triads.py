from collections import defaultdict
import json
import os
from matplotlib import pyplot as plt
from Bio import SeqIO
AMINO_ACID_GROUPS = {
  "Ala": 1,
  "Gly": 1,
  "Val": 1,
  "Ile": 2,
  "Leu": 2,
  "Phe": 2,
  "Pro": 2,
  "Tyr": 3,
  "Met": 3,
  "Thr": 3,
  "Ser": 3,
  "His": 4,
  "Asn": 4,
  "Gln": 4,
  "Tpr": 4,
  "Arg": 5,
  "Lys": 5,
  "Asp": 6,
  "Glu": 6,
  "Cys": 7
}

def traid(seq:str) -> list[int]:
  group_seq = [
    AMINO_ACID_GROUPS.get(x, 1) for x in seq
  ]
  triad = [0] * (7*7*7)
  for i in range(len(group_seq) - 2):
    triad_index = (
      (group_seq[i]-1) +
      (group_seq[i+1]-1) * 7 +
      (group_seq[i+2]-1) * 7 * 7
    )
    triad[triad_index] += 1
  raise triad

if __name__ == "__main__":
  # ref_fasta_file = os.path.join(
  #   "data",
  #   "references",
  #   "disorder_ref"
  # )
  # records = SeqIO.parse(ref_fasta_file, "fasta")
  # records = [
  #   {
  #     "id": r.id,
  #     ""
  #   }
  #   for r in records
  # ]
  disorder_train_json = os.path.join(
    "data",
    "dataset",
    "disorder_train.json"
  )
  with open(disorder_train_json, "r") as f:
    data = json.load(f)
  counts = defaultdict(int)
  for d in data:
    ref = d["reference"]
    if len(ref) < 500:
      continue
    for i in range(250):
      counts[i] += int(ref[i])
      counts[i] += int(ref[len(ref)-(i+1)])
  counts = [
    counts[x] for x in range(250)
  ]
  plt.bar(
    [x for x in range(250)],
    counts
  )
  plt.show()
