#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#define PRECISION 0.000001
#define RANGESIZE 1

void initializeMatrixes(int* A, int* B, int* C, int DIMENSION, int RAND_RANGE) {
  for (int i=  0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
			int offset = i * DIMENSION + j;
            A[offset] = (rand() % (RAND_RANGE + 1));
            B[offset] = (rand() % (RAND_RANGE + 1));
            C[offset] = 0;
		}
	}
}

void clearResults(int* C, int DIMENSION) {
  for (int i = 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
			int offset = i * DIMENSION + j;
            C[offset] = 0;
		}
	}
}

void multiplyMatrixes_1(int* A, int* B, int* C, int DIMENSION) {
    double elapsed;
    struct timeval tv1, tv2;
    struct timezone tz;
    int i, j, k;
    gettimeofday(&tv1, &tz);
	
    #pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            for (k = 0; k < DIMENSION; k++) {
				int offset = i * DIMENSION + j;
				int index1 = i * DIMENSION + k;
				int index2 = k * DIMENSION + j;
                C[offset] += A[index1] * B[index2];
            }
        }
    }

    gettimeofday(&tv2, &tz);
	
    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("Algorithm 1 - elapsed time = %f microseconds.\n", elapsed);
}

void multiplyMatrixes_2(int* A, int* B, int* C, int DIMENSION) {
    double elapsed;
    struct timeval tv1, tv2;
    struct timezone tz;
    int i, j, k;
    memset(C, 0, sizeof C);
    gettimeofday(&tv1, &tz);
	
    #pragma omp parallel for schedule(static,50) collapse(2) private(i,j,k)shared(A, B, C)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            for (k = 0; k < DIMENSION; k++) {
				int offset = i * DIMENSION + j;
				int index1 = i * DIMENSION + k;
				int index2 = k * DIMENSION + j;
                C[offset] += A[index1] * B[index2];
            }
        }
    }

    gettimeofday(&tv2, &tz);
	
    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("Algorithm 2 - elapsed time = %f microseconds.\n", elapsed);
}

void multiplyMatrixes_3(int* A, int* B, int* C, int DIMENSION) {
    double elapsed;
    struct timeval tv1, tv2;
    struct timezone tz;
    int i, j, k;
    memset(C, 0, sizeof C);
    gettimeofday(&tv1, &tz);
	
    #pragma omp parallel for schedule(dynamic,50) collapse(2) private(i,j,k) shared(A, B, C)
    for (i = 0; i < DIMENSION; i++) {
        for (j = 0; j < DIMENSION; j++) {
            for (k = 0; k < DIMENSION; k++) {
				int offset = i * DIMENSION + j;
				int index1 = i * DIMENSION + k;
				int index2 = k * DIMENSION + j;
                C[offset] += A[index1] * B[index2];
            }
        }
    }

    gettimeofday(&tv2, &tz);
	
    elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("Algorithm 3 - elapsed time = %f microseconds.\n", elapsed);
}
void printMatrixes(int* A, int* B, int* C, int DIMENSION) {
    int offset;
	printf("First input matrix \n");
    for (int i = 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
			offset = i * DIMENSION + j;
            printf("%d\t",A[offset]);
        }
        printf("\n");
    }
	
	printf("Second input matrix \n");
	for (int i = 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
			offset = i * DIMENSION + j;
            printf("%d\t",B[offset]);
        }
        printf("\n");
    }
	
	printf("Result \n");
	for (int i= 0; i < DIMENSION; i++) {
        for (int j = 0; j < DIMENSION; j++) {
			offset = i * DIMENSION + j;
            printf("%d\t",C[offset]);
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

	int *A = (int *)malloc(DIMENSION * DIMENSION * sizeof(int));
	int *B = (int *)malloc(DIMENSION * DIMENSION * sizeof(int));
	int *C = (int *)malloc(DIMENSION * DIMENSION * sizeof(int));
	
	initializeMatrixes(A, B, C, DIMENSION, RAND_RANGE);
    multiplyMatrixes_1(A, B, C, DIMENSION);
    //printMatrixes(A, B, C, DIMENSION);

    clearResults(C, DIMENSION);
    multiplyMatrixes_2(A, B, C, DIMENSION);
    //printMatrixes(A, B, C, DIMENSION);

    clearResults(C, DIMENSION);
    multiplyMatrixes_3(A, B, C, DIMENSION);
    //printMatrixes(A, B, C, DIMENSION);
    
    free(A);
    free(B);
    free(C);
	
}
