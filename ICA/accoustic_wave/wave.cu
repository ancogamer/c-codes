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
 

__constant__ float dxSquaredGPU = DX * DX;
__constant__ float dySquaredGPU = DY * DY;
__constant__ float dtSquaredGPU = DT * DT;
//--------------------------------------------------------------------------------------
// https://stackoverflow.com/questions/19646256/cudamemcpytosymbol-use-details
#define CUDA_CHECK_RETURN(value) {                                      \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                                                            \
} }
//--------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------
__global__ void wavekernel(float *prev_baseGPU,float * vel_baseGPU,float *next_baseGPU,int qtd_rows,int qtd_cols)
{    
    int  c = blockIdx.x * blockDim.x + threadIdx.x + HALF_LENGTH;
    int  r = blockIdx.y * blockDim.y + threadIdx.y + HALF_LENGTH;
    if (c < qtd_cols - HALF_LENGTH && r < qtd_rows - HALF_LENGTH) {
        int idx = r * qtd_cols + c;
        int doisPrevBaseIdx = 2.0 * prev_baseGPU[idx];
        float value = (prev_baseGPU[idx + 1] - doisPrevBaseIdx + prev_baseGPU[idx - 1]) / dxSquaredGPU;
        value += (prev_baseGPU[idx + qtd_cols] - doisPrevBaseIdx + prev_baseGPU[idx - qtd_cols]) / dySquaredGPU;      
        value *= dtSquaredGPU * vel_baseGPU[idx];      
        next_baseGPU[idx] = doisPrevBaseIdx - next_baseGPU[idx] + value;
    }
   
}
//-------------------------------------------------------------------------------------- 
void procGPU(int iterations, int rows, int cols,float *prev_base,float * vel_base,float *next_base){    
/* 
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(dxSquaredGPU, &dxSquared, 0,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(dySquaredGPU, &dySquared, 0,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(dtSquaredGPU, &dtSquared, 0,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(qtd_rows, &rows,0,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(qtd_cols, &cols,0,cudaMemcpyHostToDevice));
*/    
    
    //--------------------------------------------------------------------------------------
    float *prev_baseGPU;
    float *next_baseGPU;
    float *vel_baseGPU;
    //--------------------------------------------------------------------------------------
    cudaMalloc( (void**) &prev_baseGPU,rows * cols *sizeof(float));
    cudaMalloc( (void**) &next_baseGPU,rows * cols *sizeof(float));
    cudaMalloc( (void**) &vel_baseGPU, rows * cols *sizeof(float));
    //--------------------------------------------------------------------------------------
    cudaMemcpy(prev_baseGPU, prev_base,rows * cols *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(next_baseGPU, next_base,rows * cols *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vel_baseGPU,  vel_base, rows * cols *sizeof(float), cudaMemcpyHostToDevice);
    //--------------------------------------------------------------------------------------
    dim3 bloco = dim3(NUM_THREADS_BLOCK_X, NUM_THREADS_BLOCK_Y);
    dim3 grid = dim3(ceil ((float)cols*sizeof(float)/ (float) NUM_THREADS_BLOCK_X), ceil ((float)cols*sizeof(float)/ (float) NUM_THREADS_BLOCK_Y));
    //--------------------------------------------------------------------------------------
    printf("GRID X%u\n",grid.x);
    printf("GRID Y%u\n",grid.y);  
    float *swapGPU;
    //--------------------------------------------------------------------------------------
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
        CUDA_CHECK_RETURN( cudaEventCreate(&start) );
        CUDA_CHECK_RETURN( cudaEventCreate(&stop) );
        CUDA_CHECK_RETURN( cudaEventRecord(start, 0) );
    // launch kernal
    
    for(int n = 0; n < iterations; n++) {
        wavekernel<<<grid,bloco>>>(prev_baseGPU,vel_baseGPU,next_baseGPU,rows,cols);
        swapGPU = next_baseGPU;
        next_baseGPU = prev_baseGPU;
        prev_baseGPU = swapGPU;
    }

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    CUDA_CHECK_RETURN( error );
    
    //Obtém o erro de lançamento de kernel
    CUDA_CHECK_RETURN( cudaEventRecord(stop, 0) );
    CUDA_CHECK_RETURN( cudaEventSynchronize(stop) );
    CUDA_CHECK_RETURN( cudaEventElapsedTime(&gpu_time, start, stop) );

    //--------------------------------------------------------------------------------------
    cudaMemcpy(next_base, next_baseGPU, rows * cols *sizeof(float), cudaMemcpyDeviceToHost);
    //--------------------------------------------------------------------------------------
    // limpando o que não vai ser mais usado
    cudaFree(prev_baseGPU);cudaFree(next_baseGPU);cudaFree(vel_baseGPU);
    //--------------------------------------------------------------------------------------
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

    system("echo 'Modelo Processador: '|cat /proc/cpuinfo|grep 'model name'|head -1");
    printf("Modelo do Device: %s\n",prop.name);
    printf("Número de SMs: %d\n",prop.multiProcessorCount);
    printf("Número de Regs por SM: %d K\n",prop.regsPerMultiprocessor >> 10);
    printf("Número de Regs por Bloco: %d K\n",prop.regsPerBlock  >> 10);
    printf("Memória compartilhada por SM: %lu KB\n",prop.sharedMemPerMultiprocessor >> 10);
    printf("Memória compartilhada por Bloco: %lu KB\n",prop.sharedMemPerBlock  >> 10);
    printf("Memória Global: %lu GB\n",prop.totalGlobalMem  >> 10  >> 10  >> 10 );
    printf("Memória Constante: %lu KB\n",prop.totalConstMem  >> 10);

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
    printf("MATRIZ SIZE %lu\n",rows * cols *sizeof(float));
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
    printf("Computing wavefield ... \n");
    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);
    // wavefield modeling
    procGPU(iterations,rows,cols,prev_base,vel_base, next_base);
    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;
    save_grid(rows, cols, next_base);
    printf("Iterations completed in %f seconds \n", exec_time);

    free(prev_base);
    free(next_base);
    free(vel_base);

    return 0;
}

