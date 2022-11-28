#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#define BLOCK_SIZE 128
#define LOOP_NUMBER 5
#define min(x,y) (((x) < (y)) ? (x) : (y))

void initializeMatrixes(int** A, int** B, int** C, int DIMENSION, int RAND_RANGE, int THREAD_NUM) {
    int i, j;
    #pragma omp parallel for private(i, j) shared(A, B, C) num_threads(THREAD_NUM)
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
    #pragma omp parallel for private(i, j) shared(C) num_threads(THREAD_NUM)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            C[i][j] = 0;
        }
    }
}

void convertToOneDimension(int DIMENSION, int THREAD_NUM, int* vA, int* vB, int** A, int** B) {
    int i, j;
	#pragma omp parallel for private(i, j) shared(A, B, vA, vB) num_threads(THREAD_NUM)
	for(i = 0; i < DIMENSION; i++) {
		for(j = 0; j < DIMENSION; j++) {
			vA[i * DIMENSION + j] = A[i][j];
			vB[j * DIMENSION + i] = B[i][j];
		}
	}
}

double multiplyMatrixes_1(int** A, int** B, int** C, int DIMENSION, int THREAD_NUM) {
    double elapsed;
    int i, j;
    double start, end;

    start = omp_get_wtime();
    
    # pragma omp parallel shared(A, B, C) private(i, j) num_threads(THREAD_NUM) 
    {
        # pragma omp single
        {
            for (i = 0; i < DIMENSION; i++) {
                for (j = 0; j < DIMENSION; j++) {
                    # pragma omp task firstprivate(i, j)
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
	
    elapsed = (double) (end - start) + (double) (end - start) * 1.e-6;
    printf("Algorithm 1 - elapsed time = %f seconds.\n", elapsed);
    return elapsed;
}

double multiplyMatrixes_2(int** A, int** B, int** C, int DIMENSION, int THREAD_NUM) {
    double elapsed;
    int i, j, k;
    double start, end;

    start = omp_get_wtime();
	
    #pragma omp parallel for private(i, j, k) shared(A, B, C) num_threads(THREAD_NUM)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            for (k = 0; k < DIMENSION; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end = omp_get_wtime();
	
    elapsed = (double) (end - start) + (double) (end - start) * 1.e-6;
    printf("Algorithm 2 - elapsed time = %f seconds.\n", elapsed);
    return elapsed;
}

double multiplyMatrixes_3(int** A, int** B, int** C, int DIMENSION, int THREAD_NUM) {
    int i, j, k, ii, jj, kk;
    double elapsed;
    double start, end;
    int *vA = (int *)malloc(DIMENSION * DIMENSION * sizeof(int));
    int *vB = (int *)malloc(DIMENSION * DIMENSION * sizeof(int));
        
    start = omp_get_wtime();

    convertToOneDimension(DIMENSION, THREAD_NUM, vA, vB, A, B);

    #pragma omp parallel for private(i, j, k, ii, jj, kk) shared(C, vA, vB) num_threads(THREAD_NUM)
	for(ii = 0; ii < DIMENSION; ii += BLOCK_SIZE)
		for(jj = 0; jj < DIMENSION; jj += BLOCK_SIZE)
			for(kk = 0; kk < DIMENSION; kk += BLOCK_SIZE)
				for(i = ii; i < min(DIMENSION, ii + BLOCK_SIZE); i++)
					for(j = jj; j < min(DIMENSION, jj + BLOCK_SIZE); j++) {
						for(k = kk; k < min(DIMENSION, kk + BLOCK_SIZE); k++)
							C[i][j] += vA[i * DIMENSION + k] * vB[j * DIMENSION + k];
					}

    end = omp_get_wtime();

    free(vA);
    free(vB);

    elapsed = (double) (end - start) + (double) (end - start) * 1.e-6;
    printf("Algorithm 3 - elapsed time = %f seconds.\n", elapsed);
    return elapsed;
}

void printMatrixes(int** A, int** B, int** C, int DIMENSION) {
	printf("First input matrix \n");
    for (int i = 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            printf("%d\t", A[i][j]);
        }
        printf("\n");
    }
	
	printf("Second input matrix \n");
	for (int i = 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            printf("%d\t", B[i][j]);
        }
        printf("\n");
    }
	
	printf("Result \n");
	for (int i= 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            printf("%d\t", C[i][j]);
        }
        printf("\n");
    }
}


int main(int argc,char **argv) {
	
	// [start] [stop] [marker] [n_thr]
	
	srand(time(0));

	Args ins__args;
	parseArgs(&ins__args, &argc, argv);

	//program input argument
	int DIMENSION = ins__args.start; 
	int RAND_RANGE = ins__args.stop;
    int THREAD_NUM = ins__args.n_thr;
    double averageTime_1 = 0, averageTime_2 = 0, averageTime_3 = 0;

    for(int index = 0; index < LOOP_NUMBER; index++) {
        printf("Iteration: %d \n", index);

        int **A = (int **)malloc(DIMENSION * sizeof(int*));
        int **B = (int **)malloc(DIMENSION * sizeof(int*));
        int **C = (int **)malloc(DIMENSION * sizeof(int*));

        for (int i = 0; i < DIMENSION; i++) {
            A[i] = (int *)malloc(DIMENSION * sizeof(int));
            B[i] = (int *)malloc(DIMENSION * sizeof(int));
            C[i] = (int *)malloc(DIMENSION * sizeof(int));
        }
        
        initializeMatrixes(A, B, C, DIMENSION, RAND_RANGE, THREAD_NUM);
        averageTime_1 += multiplyMatrixes_1(A, B, C, DIMENSION, THREAD_NUM);
        //printMatrixes(A, B, C, DIMENSION);

        clearResults(C, DIMENSION, THREAD_NUM);
        averageTime_2 += multiplyMatrixes_2(A, B, C, DIMENSION, THREAD_NUM);
        //printMatrixes(A, B, C, DIMENSION);

        clearResults(C, DIMENSION, THREAD_NUM);
        averageTime_3 += multiplyMatrixes_3(A, B, C, DIMENSION, THREAD_NUM);
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
    printf("\n");
    printf("Algorithm 1 - average elapsed time = %f seconds.\n", (double) (averageTime_1 / LOOP_NUMBER) );
    printf("Algorithm 2 - average elapsed time = %f seconds.\n", (double) (averageTime_2 / LOOP_NUMBER));
    printf("Algorithm 3 - average elapsed time = %f seconds.\n", (double) (averageTime_3 / LOOP_NUMBER));
}
