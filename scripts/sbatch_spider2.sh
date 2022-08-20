#!/bin/bash -i
#SBATCH --job-name=spider2             # Job name
#SBATCH --ntasks=1                     # Run on a single CPU
#SBATCH --mem=5G                       # Requested memory
#SBATCH --time=00:10:00                # Time limit hrs:min:sec
#SBATCH --output=out/spider2_%j.log    # Standard output and error log
#SBATCH --partition=ultra,long

echo "------------------------------------------------------------------------"
echo "Job started on $(date)"
echo "------------------------------------------------------------------------"

echo Parameters: "$file_name"

seq_id="${file_name%.*}"

module load python

OUT_DIR=$(pwd)/features
PSSM_DIR=$(pwd)/pssm
SPIDER2_DIR=/projects/CAID2/caid2_dataset/SPIDER2
TEMPD=$(mktemp -d)

cd "$TEMPD" || exit

python $SPIDER2_DIR/misc/pred_pssm.py "$PSSM_DIR"/"$seq_id".pssm 2>/dev/null
$SPIDER2_DIR/HSE/run1.sh "$PSSM_DIR" "$seq_id".spd3
cp "$seq_id".spd3 "$OUT_DIR"/spd3
cp "$seq_id".hsa2 "$OUT_DIR"/hsa2
cp "$seq_id".hsb2 "$OUT_DIR"/hsb2

rm -rf "$TEMPD"

echo "------------------------------------------------------------------------"
echo "Job ended on $(date)"
echo "------------------------------------------------------------------------"

# Consider the /db/blastdb/uniref folder has been created manually in the 3 blast machines
# sbatch --job-name="spider2_test" --export file_name="DP03321.fasta" sbatch_spider2.sh
# for fasta_file in fastas/*.fasta; do fasta_file=$(basename "$fasta_file") && seq_id="${fasta_file%.*}" && sbatch --job-name="spider2_$seq_id" --export file_name="$fasta_file" sbatch_spider2.sh; done
