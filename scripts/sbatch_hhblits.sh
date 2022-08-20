#!/bin/bash -i
#SBATCH --job-name=hhblits_disprot     # Job name
#SBATCH --ntasks=4                     # Run on a single CPU
#SBATCH --mem=7G                      # Requested memory
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --output=out/hhblits_%j.log        # Standard output and error log
#SBATCH --partition=ultra,long

echo "------------------------------------------------------------------------"
echo "Job started on $(date)"
echo "------------------------------------------------------------------------"

echo Parameters: $file_name

seq_id="${file_name%.*}"

module load hhsuite

  -i fastas/$file_name -ohhm features/hhm/$seq_id.hhm -oa3m features/a3m/$seq_id.a3m -o features/hhr/$seq_id.hhr \
    -d /local/blastdb/uniclust30_2017_10/uniclust30_2017_10 -v 0 -maxres 40000 -cpu 8 -Z 0

echo "------------------------------------------------------------------------"
echo "Job ended on $(date)"
echo "------------------------------------------------------------------------"

# Consider the /db/blastdb/uniref folder has been created manually in the 3 blast machines
# sbatch --job-name="hhblits_test" --export file_name="DP03321.fasta" sbatch_hhblitst.sh
# for fasta_file in fastas/*.fasta; do fasta_file=$(basename "$fasta_file") && seq_id="${fasta_file%.*}" && sbatch --job-name="hhblits_$seq_id" --export file_name="$fasta_file" sbatch_hhblits.sh; done
