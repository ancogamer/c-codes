/*
    This program solves Laplace's equation on a regular 2D grid using simple Jacobi iteration.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>

#define ITER_MAX 3000 // number of maximum iterations
#define CONV_THRESHOLD 1.0e-5f // threshold of convergence

int itensPerThread;
int num_threads;

// matrix to be solved
double **grid;

// auxiliary matrix
double **new_grid;

// size of each side of the grid
int size;

// return the maximum value
double max(double a, double b){
    if(a > b)
        return a;
    return b;
}

// return the absolute value of a number
double absolute(double num){
    if(num < 0)
        return -1.0 * num;
    return num;
}


// initialize the grid
void initialize_grid(){
    // seed for random generator
    srand(10);
    int linf = size / 2;
    int lsup = linf + size / 10;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            // inicializa região de calor no centro do grid
            if ( i>=linf && i < lsup && j>=linf && j<lsup)
                grid[i][j] = 100;
            else
               grid[i][j] = 0;
            new_grid[i][j] = 0.0;
        }
    }
}

// save the grid in a file
void save_grid(){

    char file_name[30];
    sprintf(file_name, "grid_laplace.txt");

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            fprintf(file, "%lf ", grid[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void *allocateMemory(void *args){     
    int id = *(int *) args;
   
    int begin = id * itensPerThread;
    int end = begin + itensPerThread;
    
    if( id == num_threads-1)
       end = size; 

    for(int i=begin;i<end;i++){
        grid[i] = (double *) malloc(size * sizeof(double));
        new_grid[i] = (double *) malloc(size * sizeof(double));
    }

}
double err = 1.0;

void *calcGrid(void *args){  
    // thread id
    int id = *(int *) args;
    // calcute begin and end step of the thread
    int begin = id * itensPerThread;
    int end = begin + itensPerThread;

    if( id == num_threads-1 )
        end = size-1; 
   
    if (id == 0 )
        begin=1;
    
    for( int i = begin; i < end; i++) {
        for(int j = begin; j < end; j++) {
            new_grid[i][j] = 0.25 * (grid[i][j+1] + grid[i][j-1] +
                                     grid[i-1][j] + grid[i+1][j]);
            err = max(err, absolute(new_grid[i][j] - grid[i][j]));
        }
    }

}

int main(int argc, char *argv[]){

    if(argc < 2){
        printf("Usage: ./laplace_seq N\n");
        printf("N: The size of each side of the domain (grid)\n");
        exit(-1);
    }

    // variables to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    size = atoi(argv[1]);

    num_threads=atoi(argv[2]);

    // allocate memory to the grid (matrix)
    // creating a array of pthread_t (struct)
    pthread_t threads[num_threads];

    grid = (double **) malloc(size * sizeof(double *));
    new_grid = (double **) malloc(size * sizeof(double *)); 

    // creating a array to pass args to the threads
    itensPerThread=size/num_threads;

    int thread_args[num_threads];
    
    for (int id=0;id<num_threads;id++){
        thread_args[id]=id;
        pthread_create(&threads[id], NULL, allocateMemory, (void *) &thread_args[id]);
    }   
    
    // for(int i = 0; i < size; i++){
    for(int i = 0; i <num_threads; i++){
        pthread_join(threads[i], NULL);
    }


    // set grid initial conditions
    initialize_grid();

    int iter = 0;

    printf("Jacobi relaxation calculation: %d x %d grid\n", size, size);

    // get the start time
    gettimeofday(&time_start, NULL);

    // Jacobi iteration
    // This loop will end if either the maximum change reaches below a set threshold (convergence)
    // or a fixed number of maximum iterations have completed
    while ( err > CONV_THRESHOLD && iter <= ITER_MAX ) {

        err = 0.0;
        
        
        // calculates the Laplace equation to determine each cell's next value
        for (int i=0;i<num_threads;i++){
            thread_args[i]=i;
            pthread_create(&threads[i], NULL, calcGrid, (void *) &thread_args[i]);
        }  
        for(int i = 0; i <num_threads; i++){
            pthread_join(threads[i], NULL);
        }

        // copie the next values into the working array for the next iteration
        for( int i = 1; i < size-1; i++) {
            for( int j = 1; j < size-1; j++) {
                grid[i][j] = new_grid[i][j];
            }
        }

        if(iter % 100 == 0)
            printf("Error of %0.10lf at iteration %d\n", err, iter);

        iter++;
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                       (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    //save the final grid in file
    save_grid();

    printf("\nKernel executed in %lf seconds with %d iterations and error of %0.10lf\n", exec_time, iter, err);

    return 0;
}
