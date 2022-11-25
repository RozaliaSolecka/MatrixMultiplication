#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>

void initializeMatrixes(int** A, int** B, int** C, int DIMENSION, int RAND_RANGE, int THREAD_NUM) {
    int i, j;
    omp_set_num_threads(THREAD_NUM);
    #pragma omp parallel for private(i,j) shared(A,B,C)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            A[i][j] = (rand() % (RAND_RANGE + 1));
            B[i][j] = (rand() % (RAND_RANGE + 1));
            C[i][j] = 0;
        }
    }
}

void clearResults(int** C, int DIMENSION, int THREAD_NUM) {
    int i, j;
    omp_set_num_threads(THREAD_NUM);
    #pragma omp parallel for private(i,j) shared(C)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            C[i][j] = 0;
        }
    }
}

void multiplyMatrixes_1(int** A, int** B, int** C, int DIMENSION, int THREAD_NUM) {
    double elapsed;
    struct timeval tv1, tv2;
    struct timezone tz;
    int i, j, k;
    double start, end;
    omp_set_num_threads(THREAD_NUM);
    start = omp_get_wtime();
	
    #pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            for (k = 0; k < DIMENSION; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end = omp_get_wtime();
	
    elapsed = (double) (end-start) + (double) (end-start) * 1.e-6;
    printf("Algorithm 2 - elapsed time = %f seconds.\n", elapsed);
}

void multiplyMatrixes_2(int** A, int** B, int** C, int DIMENSION, int THREAD_NUM) {
    double elapsed;
    struct timeval tv1, tv2;
    struct timezone tz;
    int i, j;
    double start, end;
    
    # pragma omp parallel num_threads(THREAD_NUM) default(none) shared(A,B,C,start) private(i,j) firstprivate(DIMENSION)
    {
        # pragma omp single
        {
        start = omp_get_wtime();
        }
        # pragma omp single
        {
            for (i = 0; i < DIMENSION; i++) {
                for (j = 0; j < DIMENSION; j++) {
                    # pragma omp task firstprivate(i,j)
                    {
                        int k;
                        for (k = 0; k < DIMENSION; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }

    end = omp_get_wtime();
	
    elapsed = (double) (end-start) + (double) (end-start) * 1.e-6;
    printf("Algorithm 1 - elapsed time = %f seconds.\n", elapsed);
}

void printMatrixes(int** A, int** B, int** C, int DIMENSION) {
	printf("First input matrix \n");
    for (int i = 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            printf("%d\t",A[i][j]);
        }
        printf("\n");
    }
	
	printf("Second input matrix \n");
	for (int i = 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            printf("%d\t",B[i][j]);
        }
        printf("\n");
    }
	
	printf("Result \n");
	for (int i= 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            printf("%d\t",C[i][j]);
        }
        printf("\n");
    }
}


int main(int argc,char **argv) {
	
	// [start] [stop] [marker] [n_thr]
	
	srand(time(0));

	Args ins__args;
	parseArgs(&ins__args, &argc, argv);

	//set number of threads
	omp_set_num_threads(ins__args.n_thr);

	//program input argument
	int DIMENSION = ins__args.start; 
	int RAND_RANGE = ins__args.stop;
    int THREAD_NUM = ins__args.n_thr;

	int **A = (int **)malloc(DIMENSION * sizeof(int*));
	int **B = (int **)malloc(DIMENSION * sizeof(int*));
	int **C = (int **)malloc(DIMENSION * sizeof(int*));

    for (int i = 0; i < DIMENSION; i++) {
        A[i] = (int *)malloc(DIMENSION * sizeof(int));
        B[i] = (int *)malloc(DIMENSION * sizeof(int));
        C[i] = (int *)malloc(DIMENSION * sizeof(int));
    }
	
	initializeMatrixes(A, B, C, DIMENSION, RAND_RANGE, THREAD_NUM);
    multiplyMatrixes_1(A, B, C, DIMENSION, THREAD_NUM);
    //printMatrixes(A, B, C, DIMENSION);

    clearResults(C, DIMENSION, THREAD_NUM);
    multiplyMatrixes_2(A, B, C, DIMENSION, THREAD_NUM);
    //printMatrixes(A, B, C, DIMENSION);

    for (int i = 0; i < DIMENSION; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    
    free(A);
    free(B);
    free(C);
	
}
