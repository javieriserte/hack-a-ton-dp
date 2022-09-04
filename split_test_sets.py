import json
import os
from typing import Any


def parse_blast_output():
  """
  Parse the output of the BLAST program.
  """
  infile = os.path.join(
    "blasting",
    "test_results.txt"
  )
  results = []
  with open(infile, "r", encoding="utf8") as f_in:
    for line in f_in:
      line = line.strip()
      if line.startswith("# BLASTP"):
        results.append([None, None])
      if line.startswith("# Query"):
        c_id = line.split(" ")[2]
        results[-1][0] = c_id
      if line.endswith("hits found"):
        n_hits = int(line.split(" ")[1])
        results[-1][1] = n_hits
  with_hits_id = []
  without_hits_id = []
  for r in results:
    if r[1] == 0:
      without_hits_id.append(r[0])
    else:
      with_hits_id.append(r[0])
  return with_hits_id, without_hits_id

def load_test_set():
  dataset_test_json_file = os.path.join(
    "data",
    "dataset",
    "disorder_test.json"
  )
  with open(dataset_test_json_file, "r", encoding="utf8") as json_in:
    return json.load(json_in)

def subset_test(
      json_data: list[dict[str, Any]],
      wanted_ids: list[str]
    ) -> list[dict[str, Any]]:
  keep = [
    x
    for x in json_data
    if x.get('id') in wanted_ids
  ]
  return keep

def export_test(
      json_data: list[dict[str, Any]],
      title: str
    ):
  outfile = os.path.join(
    "data",
    "dataset",
    f"{title}.json"
  )
  with open(outfile, "w", encoding="utf8") as json_out:
    json.dump(
      obj = json_data,
      fp = json_out,
      indent = 2,
      ensure_ascii = False
    )


if __name__ == '__main__':
  with_hits_id, without_hits_id = parse_blast_output()
  complete_test = load_test_set()
  with_hits_test = subset_test(complete_test, with_hits_id)
  without_hits_test = subset_test(complete_test, without_hits_id)
  export_test(with_hits_test, "test_similar")
  export_test(without_hits_test, "test_dissimilar")
