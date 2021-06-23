#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

#define DT 0.0070710676f // delta t
#define DX 15.0f // delta x
#define DY 15.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s
#define HALF_LENGTH 1 // radius of the stencil

#define NUM_THREADS_BLOCK_X 32
#define NUM_THREADS_BLOCK_Y 32
 

__constant__ float dxSquared = DX * DX;
__constant__ float dySquared = DY * DY;
__constant__ float dtSquared = DT * DT;

//cudaMemcpyToSymbol(c_ny, &ny, sizeof(int)));
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}
/*
 * save the matrix on a file.txt
 */

void save_grid(int rows, int cols, float *matrix){
    system("mkdir -p wavefield");

    char file_name[64];
    sprintf(file_name, "wavefield/wavefield.txt");

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = 0; i < rows; i++) {

        int offset = i * cols;

        for(int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[offset + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    
    system("python3 plot.py");
}

void procCPU(int iterations, int rows, int cols,float *prev_base,float * vel_base,float *next_base){
    float *swap;
    for(int n = 0; n < iterations; n++) {
        for(int i = HALF_LENGTH; i < rows - HALF_LENGTH; i++) {
            for(int j = HALF_LENGTH; j < cols - HALF_LENGTH; j++) {
                // index of the current point in the grid
                int current = i * cols + j;
                
                //neighbors in the horizontal direction
                float value = (prev_base[current + 1] - 2.0 * prev_base[current] + prev_base[current - 1]) / dxSquared;
                
                //neighbors in the vertical direction
                value += (prev_base[current + cols] - 2.0 * prev_base[current] + prev_base[current - cols]) / dySquared;
                
                value *= dtSquared * vel_base[current];
                
                next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;
            }
        }

        // swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;
    }
}


__global__ void wavekernel(int iterations,float *prev_base,float * vel_base,float *next_base){
    float dxSquared = DX * DX;
    float dySquared = DY * DY;
    float dtSquared = DT * DT;

    int  c = blockIdx.x * blockDim.x + threadIdx.x- HALF_LENGTH;
    int  r = blockIdx.y * blockDim.y + threadIdx.y- HALF_LENGTH;

    float dois=2.0; 
    int idx = r * c;
    float value = (prev_base[idx + 1] - dois * prev_base[idx] + prev_base[idx - 1]) / dxSquared;
    //neighbors in the vertical direction
    value += (prev_base[idx + c] - dois * prev_base[idx] + prev_base[idx - c]) / dySquared;
    value *= dtSquared * vel_base[idx];
    next_base[idx] = dois * prev_base[idx] - next_base[idx] + value;
   
}
 
void procGPU(int iterations, int rows, int cols,float *prev_base,float * vel_base,float *next_base){
    //Aloca espaço na CPU para o resultado
    // alocando o tamanho da matriz  
    // ponteiros para gpu
    unsigned long matrizSize = rows * cols * sizeof(float);
    float *prev_baseGPU;
    float *next_baseGPU;
    float *vel_baseGPU;
    float *swapGPU;

    cudaMalloc( (void**) &prev_baseGPU,matrizSize);
    cudaMalloc( (void**) &next_baseGPU,matrizSize);
    cudaMalloc( (void**) &vel_baseGPU,matrizSize);
    //cudaMalloc( (void**) &swapGPU,rows * cols * sizeof(float));
    //-------------------------------------
    cudaMemcpy(prev_baseGPU, prev_base,matrizSize, cudaMemcpyHostToDevice);
    cudaMemcpy(next_baseGPU, next_base,matrizSize, cudaMemcpyHostToDevice);
    cudaMemcpy(vel_baseGPU, vel_base,matrizSize, cudaMemcpyHostToDevice);
 
    dim3 bloco = dim3(NUM_THREADS_BLOCK_X, NUM_THREADS_BLOCK_Y);
    dim3 grid = dim3(ceil (matrizSize/ (float) NUM_THREADS_BLOCK_X), ceil (matrizSize/ (float) NUM_THREADS_BLOCK_Y));
 
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
        checkCuda( cudaEventCreate(&start) );
        checkCuda( cudaEventCreate(&stop) );
        checkCuda( cudaEventRecord(start, 0) );

    for(int n = 0; n < iterations; n++) {
        // swap arrays for next iteration
        // viraram ponteiros para gpu
        //int iterations,float *prev_base,float * vel_base,float *next_base
        wavekernel<<<grid,bloco>>>(n,prev_baseGPU,vel_baseGPU,next_baseGPU);
        swapGPU = next_baseGPU;
        next_baseGPU = prev_baseGPU;
        prev_baseGPU = swapGPU;
    }

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    checkCuda( error );

    //Obtém o erro de lançamento de kernel
    checkCuda( cudaEventRecord(stop, 0) );
    checkCuda( cudaEventSynchronize(stop) );
    checkCuda( cudaEventElapsedTime(&gpu_time, start, stop) );
    //-------------------------------------------------------------
    cudaMemcpy(next_base, next_baseGPU, matrizSize, cudaMemcpyDeviceToHost);
    cudaFree(prev_baseGPU);
    cudaFree(next_baseGPU);
    cudaFree(vel_baseGPU);
    // limpando o que não vai ser mais usado
    free(prev_base);
    free(vel_base);
    //-------------------------------------------------------------
    //Imprime o resultado
        printf("Tempo de Execução na GPU: %.4f ms ", gpu_time);

}



int main(int argc, char* argv[]) {

    if(argc != 4){
        printf("Usage: ./stencil N1 N2 TIME\n");
        printf("N1 N2: grid sizes for the stencil\n");
        printf("TIME: propagation time in ms\n");
        exit(-1);
    }
        cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);

    printf("Modelo Processador%s",system("cat /proc/cpuinfo|grep 'model name'|head -1"));
    printf("Modelo do Device: %s\n",prop.name);
    printf("Número de SMs: %d\n",prop.multiProcessorCount);
    printf("Número de Regs por SM: %d K\n",prop.regsPerMultiprocessor >> 10);
    printf("Número de Regs por Bloco: %d K\n",prop.regsPerBlock  >> 10);
    printf("Memória compartilhada por SM: %d KB\n",prop.sharedMemPerMultiprocessor >> 10);
    printf("Memória compartilhada por Bloco: %d KB\n",prop.sharedMemPerBlock  >> 10);
    printf("Memória Global: %d GB\n",prop.totalGlobalMem  >> 10  >> 10  >> 10 );
    printf("Memória Constante: %d KB\n",prop.totalConstMem  >> 10);

    // number of rows of the grid
    int rows = atoi(argv[1]);

    // number of columns of the grid
    int cols = atoi(argv[2]);

    // number of timesteps
    int time = atoi(argv[3]);
    
    // calc the number of iterations (timesteps)
    int iterations = (int)((time/1000.0) / DT);

    // represent the matrix of wavefield as an array
    float *prev_base = (float*) malloc(rows * cols * sizeof(float));
    float *next_base = (float*) malloc(rows * cols * sizeof(float));

    // represent the matrix of velocities as an array
    float *vel_base =(float*) malloc(rows * cols * sizeof(float));

    printf("Grid Sizes: %d x %d\n", rows, cols);
    printf("Iterations: %d\n", iterations);

    // ************* BEGIN INITIALIZATION *************

    printf("Initializing ... \n");

    // define source wavelet
    float wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                         0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                         0.023680172, 0.005611435,  0.001823209,  -0.000720549};

    // initialize matrix
    for(int i = 0; i < rows; i++){

        int offset = i * cols;

        for(int j = 0; j < cols; j++){
            prev_base[offset + j] = 0.0f;
            next_base[offset + j] = 0.0f;
            vel_base[offset + j] = V * V;
        }
    }

    // add a source to initial wavefield as an initial condition
    for(int s = 11; s >= 0; s--){
        for(int i = rows / 2 - s; i < rows / 2 + s; i++){

            int offset = i * cols;

            for(int j = cols / 2 - s; j < cols / 2 + s; j++)
                prev_base[offset + j] = wavelet[s];
        }
    }

    // ************** END INITIALIZATION **************
    
    float *prev_base1 = (float*) malloc(rows * cols * sizeof(float));
    float *next_base1 = (float*) malloc(rows * cols * sizeof(float));
    float *vel_base1 =(float*) malloc(rows * cols * sizeof(float));

    memcpy(prev_base1, prev_base, rows * cols * sizeof(float));
    memcpy(next_base1, next_base, rows * cols * sizeof(float));
    memcpy(vel_base1, vel_base,   rows * cols * sizeof(float));

    printf("Computing wavefield ... \n");

    
    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);
    // wavefield modeling
    procCPU(iterations,rows,cols,prev_base,vel_base, next_base);

    procGPU(iterations,rows,cols,prev_base1,vel_base1, next_base1);
    
    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    save_grid(rows, cols, next_base);
    save_grid(rows, cols, next_base1);
    
    printf("Iterations completed in %f seconds \n", exec_time);

    free(prev_base);
    free(next_base);
    free(vel_base);
    free(next_base1);



    return 0;
}
