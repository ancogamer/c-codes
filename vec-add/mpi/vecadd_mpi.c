/*
    Simple program to add two vectors and verify the results.

    Based on Tim Mattson's (November 2017) implementation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define TOL 0.0000001

int main(int argc, char *argv[]) {

    if(argc != 2){
        printf("Usage: ./vecadd_mpi N\n");
        printf("N: Size of the vectors\n");
        exit(-1);
    }

    // variables to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);

    int vSize = atoi(argv[1]);
    int err = 0;
    float *a = (float*) malloc(vSize * sizeof(float));
    float *b = (float*) malloc(vSize * sizeof(float));
    float *c = (float*) malloc(vSize * sizeof(float));
    float *res = (float*) malloc(vSize * sizeof(float));
 
    int thread, np;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread);

    int num_local_steps = 0;
    int num_steps = atoi(argv[1]);
    int remainder = num_steps % (np - 1);       /* if num_steps is not divisible by np */
    if (remainder > 0)
      num_local_steps = num_steps / (np - 1);   /* number of steps per worker */
    else
      num_local_steps = num_steps / np;
    
    int start, stop;

    if (thread == 0){     /* master process */
      start = 1;
      if (remainder >0) stop = start + remainder;
      else stop = start + num_local_steps;
    } else {     /*  worker processes */
      if (remainder > 0) start = (thread - 1) * num_local_steps + remainder + 1;
      else start = thread * num_local_steps + 1;
      stop = start + num_local_steps;
    }

    
    // add two vectors
    for (int i=start;i < stop; i++){
        a[i] = (float) i;
        b[i] = 2.0 * (float) i;
        c[i] = 0.0;
        res[i] = i + 2 * i;
        c[i] = a[i] + b[i];
        //printf(" Process %d:  i=%d  a=%.2f, b=%.2f, c=%f  \n",thread,i,a[i],b[i],c[i]);
    }
    if (thread == 0){  
        for (start; start < stop; start++){
            float val = c[start] - res[start];
            val = val * val;

            if(val > TOL)
                err++;
        }

        // get the end time
        gettimeofday(&time_end, NULL);

        double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                           (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

        printf("vectors added with %d errors in %lf seconds\n", err, exec_time);

        free(a);
        free(b);
        free(c);
        free(res);
    }
    MPI_Finalize();
    return 0;
}
