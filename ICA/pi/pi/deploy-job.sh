#!/bin/bash
#SBATCH -J mmul                     # Job name
#SBATCH -p fast                     # Job partition
#SBATCH -n 1                        # Number of processes
#SBATCH -t 01:30:00                 # Run time (hh:mm:ss)
#SBATCH --cpus-per-task=40          # Number of CPUs per process
#SBATCH --output=%x.%j.out          # Name of stdout output file - %j expands to jobId and %x to jobName
#SBATCH --error=%x.%j.err           # Name of stderr output file

echo "*** SEQUENTIAL ***"
srun singularity run container.sif pi_seq 1000000000
mv result_matrix.txt pi_seq.txt

echo "*** PTHREAD ***"
srun singularity run container.sif pi_pth 1000000000 1
mv result_matrix.txt pi_pth1.txt
srun singularity run container.sif pi_pth 1000000000 2
mv result_matrix.txt pi_pth2.txt
srun singularity run container.sif pi_pth 1000000000 5
mv result_matrix.txt pi_pth5.txt
srun singularity run container.sif pi_pth 1000000000 10
mv result_matrix.txt pi_pth5.txt
srun singularity run container.sif pi_pth 1000000000 20
mv result_matrix.txt pi_pth5.txt
srun singularity run container.sif pi_pth 1000000000 40
mv result_matrix.txt pi_pth40.txt

echo "*** OPENMP ***"
export OMP_NUM_THREADS=1
srun singularity run container.sif pi_omp 1000000000
mv result_matrix.txt pi_omp.txt
export OMP_NUM_THREADS=2
srun singularity run container.sif pi_omp 1000000000
mv result_matrix.txt pi_omp.txt
export OMP_NUM_THREADS=5
srun singularity run container.sif pi_omp 1000000000
mv result_matrix.txt pi_omp.txt
export OMP_NUM_THREADS=10
srun singularity run container.sif pi_omp 1000000000
mv result_matrix.txt pi_omp.txt
export OMP_NUM_THREADS=20
srun singularity run container.sif pi_omp 1000000000
mv result_matrix.txt pi_omp.txt
export OMP_NUM_THREADS=40
srun singularity run container.sif pi_omp 1000000000
mv result_matrix.txt pi_omp.txt

diff pi_seq.txt pi_pth.txt
diff pi_seq.txt pi_omp.txt
 
