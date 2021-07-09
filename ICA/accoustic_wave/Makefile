CC=gcc

all: clean sequential

sequential:
	$(CC) wave_seq.c -o wave_seq
	./wave_seq 512 512 2000	  > ./terminal_output/seq/wave_seq_512_512_2000		&& mv ./plots/wavefield.png ./plots/512_512.png   && mv ./plots/512_512.png ./expected_imgs/
	./wave_seq 1024 1024 2000 > ./terminal_output/seq/wave_seq_1024_1024_2000	&& mv ./plots/wavefield.png ./plots/1024_1024.png && mv ./plots/1024_1024.png ./expected_imgs/  
	./wave_seq 2048 2048 2000 > ./terminal_output/seq/wave_seq_2048_2048_2000	&& mv ./plots/wavefield.png ./plots/2048_2048.png && mv ./plots/2048_2048.png ./expected_imgs/
	./wave_seq 4096 4096 2000 > ./terminal_output/seq/wave_seq_4096_4096_2000	&& mv ./plots/wavefield.png ./plots/4096_4096.png && mv ./plots/4096_4096.png ./expected_imgs/
	./wave_seq 8192 8192 2000 > ./terminal_output/seq/wave_seq_8192_8192_2000	&& mv ./plots/wavefield.png ./plots/8192_8192.png && mv ./plots/8192_8192.png ./expected_imgs/

cuda:
	nvcc wave.cu -o wave_cuda
	./wave_cuda 512 512 2000   > ./terminal_output/cuda/wave_cuda_512_512_2000   && mv ./plots/wavefield.png ./plots/512_512.png   && mv ./plots/512_512.png ./got_imgs/ 
	./wave_cuda 1024 1024 2000 > ./terminal_output/cuda/wave_cuda_1024_1024_2000 && mv ./plots/wavefield.png ./plots/1024_1024.png && mv ./plots/1024_1024.png ./got_imgs/
	./wave_cuda 2048 2048 2000 > ./terminal_output/cuda/wave_cuda_2048_2048_2000 && mv ./plots/wavefield.png ./plots/2048_2048.png && mv ./plots/2048_2048.png ./got_imgs/
	./wave_cuda 4096 4096 2000 > ./terminal_output/cuda/wave_cuda_4096_4096_2000 && mv ./plots/wavefield.png ./plots/4096_4096.png && mv ./plots/4096_4096.png ./got_imgs/
	./wave_cuda 8192 8192 2000 > ./terminal_output/cuda/wave_cuda_8192_8192_2000 && mv ./plots/wavefield.png ./plots/8192_8192.png && mv ./plots/8192_8192.png ./got_imgs/

clean:
	rm -f wave_seq wave_omp wave_mpi wave_pth wave_cuda
