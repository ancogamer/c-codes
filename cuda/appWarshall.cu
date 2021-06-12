#include <stdio.h>
#include <time.h>
#include <assert.h>

#define QTD_ELEMENTOS 1024
#define NUM_THREADS_BLOCK_X 32
#define NUM_THREADS_BLOCK_Y 32

void inicializaMatriz(int *data, unsigned size)
{
  time_t t;
	srand((unsigned int) time(&t));
	for (int i=0; i<size; i++) {
     for (int j=0; j<size; j++) {
		   //data[i * size + j] = (int)( rand() & 0xFF )/10.0f;
			 data[i * size + j] = ((int)rand() ) % 2;
     	}
	}
}


void warshallCPU(int* A, int* F, unsigned n)
{
		for(int k = 0; k < n; k++){
			for(int lin = 0; lin < n; lin ++){
				for(int col = 0; col < n; col ++){
						if(A[k * n + col] == 1 && A[lin * n + k] == 1)
							F[lin * n + col] = 1;
				}
			}
		}
}
__global__ void warshallGPU(int k,int *A,int *F, unsigned n, int *R)
{
  
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int l = blockIdx.y * blockDim.y + threadIdx.y;

  int i = c * n + l;
  __shared__ int best;
	if(threadIdx.x==0)
		best=A[n*blockIdx.y+k];
	__syncthreads();
  // não tenho numero da interação.
  if(A[k * n + c] == 1 && A[l * n + k] == 1)
		F[l * n + c] = 1;
  R[i] = A[i] + F[i];
	

}
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void imprimeSoma(int* data, unsigned n)
{
    double soma = 0;
    for (int i=0; i < n; i++) {
                for (int j=0; j < n; j++){
                    soma += data[i * n + j];        
                }        
      } 
    printf("A soma é %f\n",soma);
}
void processamentoGPU(int *A,int *B ,unsigned n){

	//Aloca espaço na CPU para o resultado
	int matrizSize = sizeof(int) * n * n;
	// alocando o tamanho da matriz
	int* F = (int*) malloc(matrizSize);

	// ponteiros para gpu

	int* gA;
	int* gB;
	int* gR;
	int* r=0;

	cudaMalloc( (void**) &gA, matrizSize);
  	cudaMalloc( (void**) &gB, matrizSize);
  	cudaMalloc( (void**) &gR, matrizSize);

	//-------------------------------------

	cudaMemcpy(gA, A, matrizSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gB, B, matrizSize, cudaMemcpyHostToDevice);

  	dim3 bloco = dim3(NUM_THREADS_BLOCK_X, NUM_THREADS_BLOCK_Y);
  	dim3 grid = dim3(ceil (n/ (float) NUM_THREADS_BLOCK_X), ceil (n/ (float) NUM_THREADS_BLOCK_Y));

	cudaEvent_t start, stop;
    float gpu_time = 0.0f;
    	checkCuda( cudaEventCreate(&start) );
    	checkCuda( cudaEventCreate(&stop) );
    	checkCuda( cudaEventRecord(start, 0) );

	for (int k =0; k<n;k++){
		warshallGPU<<<grid,bloco>>>(k,gA, gB, n,gR);
	}
	cudaDeviceSynchronize();
	//Obtém o erro de lançamento de kernel
    cudaError_t error = cudaGetLastError();
    checkCuda( error );

 	checkCuda( cudaEventRecord(stop, 0) );
    checkCuda( cudaEventSynchronize(stop) );
    checkCuda( cudaEventElapsedTime(&gpu_time, start, stop) );

	cudaMemcpy(r, gR, matrizSize, cudaMemcpyDeviceToHost);

	cudaFree(gA);
  	cudaFree(gB);
  	cudaFree(gR);

  	//Imprime o resultado
  	imprimeSoma(r, n);
   		printf("Tempo de Execução na GPU: %.4f ms ", gpu_time);



}
void processamentoCPU(int *A, unsigned n)
{
	int* F = (int*) malloc( sizeof(int) * n * n);
  	double tempoGasto;
	clock_t start = clock();
	warshallCPU(A, F, n);
	clock_t stop = clock();
	tempoGasto = 1000 *  (stop - start) / (float) CLOCKS_PER_SEC;
	printf("Tempo de execução da CPU: %f ms\n", tempoGasto );
	free(F);
}

void mainWarshall()
{

	int byteNumber = QTD_ELEMENTOS * QTD_ELEMENTOS * sizeof(int);

	int *A = (int*) malloc(byteNumber);

    int *B = (int*) malloc(byteNumber);
	inicializaMatriz(A, QTD_ELEMENTOS);
	inicializaMatriz(B, QTD_ELEMENTOS);

	
	processamentoCPU(A, QTD_ELEMENTOS);
	processamentoGPU(A,B, QTD_ELEMENTOS);
  
	free(A);
	free(B);
}

int main(void)
{

	cudaSetDevice(0);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	printf("Número de SM: %d\n",prop.multiProcessorCount);
	printf("Modelo GPU: %s\n",prop.name);

	mainWarshall();
	return 0;
}
