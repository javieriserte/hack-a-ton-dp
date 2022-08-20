#!/bin/bash -i
#SBATCH --job-name=hhblits_disprot     # Job name
#SBATCH --ntasks=8                     # Run on a single CPU
#SBATCH --mem=15G                      # Requested memory
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --output=out/psiblast_%j.log        # Standard output and error log
#SBATCH --partition=ultra,long

echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"

echo Parameters: $file_name

seq_id="${file_name%.*}"

module load ncbi-blast/latest

if [ ! -f pssm/$seq_id.pssm ]; then
	psiblast -db /local/blastdb/uniref50/uniref50 -query fastas/$file_name -out_ascii_pssm features/pssm/$seq_id.pssm -save_pssm_after_last_round -num_threads 8 -num_iterations 3 1> /dev/null 2> /dev/null
fi

echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"

# Consider the /db/blastdb/uniref folder has been created manually in the 3 blast machines
# sbatch --job-name="psiblast_test" --export file_name="DP03321.fasta" sbatch_psiblast.sh
# for fasta_file in fastas/*.fasta; do fasta_file=$(basename "$fasta_file") && seq_id="${fasta_file%.*}" && sbatch --job-name="psiblast_$seq_id" --export file_name="$fasta_file" sbatch_psiblast.sh; done 
