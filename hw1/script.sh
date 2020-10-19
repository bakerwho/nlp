#!/bin/bash
#SBATCH --job-name=nlpvecs
#SBATCH --output=/project2/jevans/aabir/nlp/hw1/script.out
#SBATCH --error=/project2/jevans/aabir/nlp/hw1/script.err
#SBATCH --nodes=1
#SBATCH --partition=broadwl
#SBATCH --mem=31GB
#SBATCH --time=32:00:00

module load python/anaconda-2020.02

echo 'run started at' $(date)
python3 /project2/jevans/aabir/nlp/hw1/get_vecs.py
echo 'run ended at' $(date)
