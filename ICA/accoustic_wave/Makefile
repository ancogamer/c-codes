CC=gcc

all: clean sequential

sequential:
	$(CC) wave_seq.c -o wave_seq
	./wave_seq 512 512 2000
cuda:
	nvcc wave.cu  -w -o wave_cuda
	./wave_cuda 512 512 2000
	
clean:
	rm -f wave_seq wave_omp wave_mpi wave_pth wave_cuda