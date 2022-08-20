#!/bin/bash -i

module load cd-hit
module load ncbi-blast/2.2.17

start_dir=$(pwd)
input_file=$1
input_file_basename=$(echo "${input_file%.*}")
tmp_dir=$(mktemp -d)
cpus=$(grep -c 'cpu[0-9]' /proc/stat)

mkdir -p clusters
cp $input_file $tmp_dir
cd $tmp_dir

echo "Clustering to 80%"
cd-hit -i $input_file -o nr80 -c 0.8 -n 5 -d 0 -M 16000 -T $cpus
echo "Clustering to 60%"
cd-hit -i nr80 -o nr60 -c 0.6 -n 4 -d 0 -M 16000 -T $cpus
echo "Clustering to 30%"
psi-cd-hit.pl -i nr60 -o nr30 -c 0.3 -exec local -core 16
echo "Merging results"
clstr_rev.pl nr80.clstr nr60.clstr >nr80-60.clstr
clstr_rev.pl nr80-60.clstr nr30.clstr >nr80-60-30.clstr

clstr2txt.pl nr80-60-30.clstr >$start_dir/clusters/$input_file_basename-30.clstr
echo "Output file $input_file_basename-30.clstr created"
echo "Finished"

rm -rf tmp_dir
