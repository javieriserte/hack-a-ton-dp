
# makeblastdb -in data/dataset/disorder_train.fasta -out blasting/train -dbtype prot

# blastp \
#   -query data/dataset/disorder_test.fasta \
#   -db blasting/train \
#   -out blasting/test_results.txt \
#   -evalue 1E-5 \
#   -outfmt 7

