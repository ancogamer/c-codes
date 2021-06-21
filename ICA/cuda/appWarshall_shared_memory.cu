#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>

#define QTD_ELEMENTOS 2048

#define NUM_THREADS_BLOCK_X 32
#define NUM_THREADS_BLOCK_Y 32
 
void inicializaMatriz(int *data, unsigned size)
{
  time_t t;
    srand((unsigned int) time(&t));
    for (int i=0; i<size; i++) {
     for (int j=0; j<size; j++) {
             data[i * size + j] = ((int)rand() ) % 2;
        }
    }
}
 
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}
 
 
void warshallCPU(int* fechoMatriz, unsigned n)
{
    for(int k = 0; k < n; k++){
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                    if(fechoMatriz[k * n + j] == 1 && fechoMatriz[i * n + k] == 1)  
                        fechoMatriz[i * n + j] = 1;
            }           
        }                   
    }
}
void imprimeSoma(int *data, unsigned n)
{
    double soma = 0;
    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++){
            soma += data[i * n + j];  
        }     
    } 

    printf("A soma é %d\n",soma);
}
 
__global__ void warshallGPU1(int *A,int k, unsigned n){
 int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int ladrilhoA[NUM_THREADS_BLOCK_Y][NUM_THREADS_BLOCK_X];
    __shared__ int ladrilhoB[NUM_THREADS_BLOCK_Y][NUM_THREADS_BLOCK_X];

    int tidX;
    int tidY;
    
    tidX = blockIdx.x * blockDim.x + threadIdx.x;
    tidY = k * blockDim.y + threadIdx.y;
    ladrilhoA[threadIdx.y][threadIdx.x] = A[tidY * n + tidX];

    tidX = k * blockDim.x + threadIdx.x;
    tidY = blockIdx.y * blockDim.y + threadIdx.y;
    ladrilhoB[threadIdx.y][threadIdx.x] = A[tidY * n + tidX];

    __syncthreads();

    for(int m = 0; m < blockDim.x; m++) {
        if (ladrilhoA[m][threadIdx.x] == 1 &&
            ladrilhoB[threadIdx.y][m] == 1
            ) {
            A[i * n + j] = 1;
        }
        __syncthreads();
    }
}
void processamentoGPU(int *A ,unsigned n){
    //Aloca espaço na CPU para o resultado
    int matrizSize = sizeof(int) * n * n;
    // alocando o tamanho da matriz
    int* F = (int*) malloc(matrizSize);
    // ponteiros para gpu
    int* gA;
    cudaMalloc( (void**) &gA, matrizSize);
    //-------------------------------------
    cudaMemcpy(gA, A, matrizSize, cudaMemcpyHostToDevice);
 
    dim3 bloco = dim3(NUM_THREADS_BLOCK_X, NUM_THREADS_BLOCK_Y); 
    
    //printf("block size %d %d \n",NUM_THREADS_BLOCK_X,NUM_THREADS_BLOCK_Y);

    dim3 grid = dim3(ceil (n/ (float) NUM_THREADS_BLOCK_X), ceil (n/ (float) NUM_THREADS_BLOCK_Y));
    //printf("grid size %d %d \n",NUM_THREADS_BLOCK_X,NUM_THREADS_BLOCK_Y);

    
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
        checkCuda( cudaEventCreate(&start) );
        checkCuda( cudaEventCreate(&stop) );
        checkCuda( cudaEventRecord(start, 0) );
 
        //for(int k = 0; k < (n / NUM_THREADS_BLOCK_X); k++){
        warshallGPU1<<<grid,bloco>>>(gA,0, n);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        checkCuda( error );
        warshallGPU1<<<grid,bloco>>>(gA,0, n);

        cudaDeviceSynchronize();
        error = cudaGetLastError();
        checkCuda( error );
        //}
       

    //Obtém o erro de lançamento de kernel
    checkCuda( cudaEventRecord(stop, 0) );
    checkCuda( cudaEventSynchronize(stop) );
    checkCuda( cudaEventElapsedTime(&gpu_time, start, stop) );
    //-------------------------------------------------------------
    cudaMemcpy(F, gA, matrizSize, cudaMemcpyDeviceToHost);
    cudaFree(gA);
    //-------------------------------------------------------------
    //Imprime o resultado
    printf("Tempo de Execução na GPU: %.4f ms \n", gpu_time);
    imprimeSoma(F, n);
    free(F);
       
}
 
void processamentoCPU(int *A, unsigned n)
{
    int* F = (int*) malloc( sizeof(int) * n * n);
    memcpy(F, A, sizeof(int)*n*n);
    double tempoGasto;
    
    clock_t start = clock();    
        warshallCPU(F, n);
    clock_t stop = clock();
    tempoGasto = (stop - start) / (float) CLOCKS_PER_SEC;
    printf("Tempo de execução da CPU: %f s\n", tempoGasto ); 
    imprimeSoma(F, n);
    free(F);
}
 
void mainWarshall()
{
 
    int byteNumber = QTD_ELEMENTOS * QTD_ELEMENTOS * sizeof(int);
 
    int *A = (int*) malloc(byteNumber);
 
    inicializaMatriz(A, QTD_ELEMENTOS);
    
    processamentoCPU(A, QTD_ELEMENTOS);
    processamentoGPU(A, QTD_ELEMENTOS);
  
    free(A);
}
 
int main(void)
{
 
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);

    system("cat /proc/cpuinfo|grep 'model name'|head -1");
    printf("Modelo do Device: %s\n",prop.name);
    printf("Número de SMs: %d\n",prop.multiProcessorCount);
    printf("Número de Regs por SM: %d K\n",prop.regsPerMultiprocessor >> 10);
    printf("Número de Regs por Bloco: %d K\n",prop.regsPerBlock  >> 10);
    printf("Memória compartilhada por SM: %d KB\n",prop.sharedMemPerMultiprocessor >> 10);
    printf("Memória compartilhada por Bloco: %d KB\n",prop.sharedMemPerBlock  >> 10);
    printf("Memória Global: %d GB\n",prop.totalGlobalMem  >> 10  >> 10  >> 10 );
    printf("Memória Constante: %d KB\n",prop.totalConstMem  >> 10);
    
    mainWarshall();
    return 0;
}