#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif


#define NUM_THREADS	64
#define DIMS 3
#define RUNS 200
#define DIM0 4096
#define	DIM1 4096
#define DIM2 4
#define MIN_DOMAIN -1.0f
#define MAX_DOMAIN 1.0f
#define DOMAIN_DELTA (1.0f / (MAX_DOMAIN - MIN_DOMAIN))
#define SIZE (DIM0 * DIM1 * DIM2)
#define BYTES (SIZE * sizeof(float))
#define TIND(x) (x * SIZE)
#define DEBUG 0


void print_firstn_array(float *a, int to_print, const int N);
void randbits_init_array(float *a, int bits, const int N);
void zero_out_array(float *a, const int N);
void init_x(float *a, const int N);
void init_y(float *a, const int N);
void init_z(float *a, const int N);
int array_equal(float *a, float *b, const int N);
__global__ void warp(float *tensors, float *image, float *result, float *dimensions);

int main() {

	// Allocate memory on host
	float *tensors = (float *)malloc(DIMS * BYTES); // tensor array
	float *image = (float *)malloc(BYTES);			// image to warp
	float *result = (float *)malloc(BYTES);			// where to store the result of the warp operator
	float dimensions[DIMS] = { DIM0, DIM1, DIM2 };	// dimensions

	printf("\nMemory allocated on host.\n");

	// Initialize tensor arrays
	
	init_x(tensors + TIND(0), SIZE);
	//printf("Tensors x done\n");
	init_y(tensors + TIND(1), SIZE);
	//printf("Tensors y done\n");
	init_z(tensors + TIND(2), SIZE);
	//printf("Tensors z done\n");
	


	printf("Tensors initialized.\n");

	// Zero out result and initialize image
	randbits_init_array(image, 8, SIZE);
	zero_out_array(result, SIZE);

	// Print initial image

	if (DEBUG) {
		printf("Image to warp:\n");
		print_firstn_array(image, SIZE, SIZE);

		printf("Tensor x:\n");
		print_firstn_array(tensors + TIND(0), SIZE, SIZE);
		printf("Tensor y:\n");
		print_firstn_array(tensors + TIND(1), SIZE, SIZE);
		printf("Tensor z:\n");
		print_firstn_array(tensors + TIND(2), SIZE, SIZE);
	}


	// Allocate memory in device
	float *dtensors, *dimage, *dresult;
	float *ddimensions;

	checkCudaErrors(cudaMalloc(&dtensors, DIMS * BYTES));
	checkCudaErrors(cudaMalloc(&dimage, BYTES));
	checkCudaErrors(cudaMalloc(&dresult, BYTES));
	checkCudaErrors(cudaMalloc(&ddimensions, DIMS * sizeof(float)));

	printf("Allocated memory on device.\n");

	// Copy data from host to device
	checkCudaErrors(cudaMemcpy(dtensors, tensors, DIMS * BYTES, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dimage, image, BYTES, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dresult, result, BYTES, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ddimensions, dimensions, DIMS * sizeof(float), cudaMemcpyHostToDevice));

	printf("Data copied to device.\n");

	// Launch kernel
	int num_blocks = SIZE / NUM_THREADS + 1;
	printf("\nNumber of blocks :\t%d\n", num_blocks);
	printf("Threads per block:\t%d\n", NUM_THREADS);
	printf("Tensor resolution:\t[%d, %d, %d]\n", DIM0, DIM1, DIM2);
	printf("Evaluation points:\t%d\n", (DIM0 * DIM1 * DIM2));

	printf("\nStarting %d warp runs...\n", RUNS);

	for(int i = 0; i < RUNS; ++i) {
		warp << < num_blocks, NUM_THREADS >> > (dtensors, dimage, dresult, ddimensions);

		// Wait for GPU to finish
		checkCudaErrors(cudaDeviceSynchronize());
	}
	printf("Done.\n\n");
	checkCudaErrors(cudaMemcpy(result, dresult, BYTES, cudaMemcpyDeviceToHost));

	// Print result tensor froim warp operator
	if (DEBUG) {
		printf("Result image:\n");
		print_firstn_array(result, SIZE, SIZE);
	}
	//printf("Warp identity: %s\n", ((array_equal(image, result, SIZE) == 0) ? "Passed" : "Failed"));
	//int problem_index = array_equal(image, result, SIZE);
	//printf("Warp identity: %d\n", problem_index);
	//printf("Problem Index: (tx, ty, tz, image, result): (%.3f, %.3f, %.3f, %.3f, %.3f)\n", tensors[problem_index], tensors[TIND(1) + problem_index], tensors[TIND(2) + problem_index], image[0], result[1]);
	// Free up memory
	//printf("Result: %.3f\n", result[0]);

	checkCudaErrors(cudaFree(dtensors));
	checkCudaErrors(cudaFree(dimage));
	checkCudaErrors(cudaFree(dresult));
	free(tensors);
	free(image);
	free(result);


	return 0;
}

int array_equal(float *a, float *b, const int N) {
	int count = 0;
	for (int i = 0; i < N; ++i) {
		if (a[i] != b[i]) {
			++count;
		}
	}
	return count;
}

void print_firstn_array(float *a, int to_print, const int N) {
	int limit = (to_print > N) ? N : ((to_print < 0) ? 0 : to_print);
	for (int i = 0; i < limit; ++i) {
		printf("%.3f\n", a[i]);
	}
}

void randbits_init_array(float *a, int bits, const int N) {
	srand(1234567890);
	int mask = (1 << bits) - 1;
	for (int i = 0; i < N; ++i) {
		a[i] = (float)(rand() & mask);
	}
}

void zero_out_array(float *a, const int N) {
	for (int i = 0; i < N; ++i) {
		a[i] = 0.0;
	}
}

void init_x(float *a, const int N) {
	for (int i = 0; i < DIM0; ++i) {
		for (int j = 0; j < DIM1; ++j) {
			for (int k = 0; k < DIM2; ++k) {
				int indices = DIM2 * DIM1 * i + DIM2 * j + k;
				a[indices] = (float)(i / ((DIM0 - 1.0) * 0.5)) - 1.0;
			}
		}
	}
}

void init_y(float *a, const int N) {
	for (int i = 0; i < DIM0; ++i) {
		for (int j = 0; j < DIM1; ++j) {
			for (int k = 0; k < DIM2; ++k) {
				int indices = DIM2 * DIM1 * i + DIM2 * j + k;
				a[indices] = (j / ((DIM1 - 1.0) * 0.5)) - 1.0;
			}
		}
	}
}

void init_z(float *a, const int N) {
	for (int i = 0; i < DIM0; ++i) {
		for (int j = 0; j < DIM1; ++j) {
			for (int k = 0; k < DIM2; ++k) {
				int indices = DIM2 * DIM1 * i + DIM2 * j + k;
				a[indices] = (k / ((DIM2 - 1) * 0.5)) - 1.0;
			}
		}
	}
}

__global__ void warp(float *tensors, float *image, float *result, float *dimensions) {
	int ind = blockIdx.x * blockDim.x + threadIdx.x;

#define __COORD(x)	lrintf(									\
		(													\
			fmaxf(											\
				fminf(tensors[TIND(x) + ind], MAX_DOMAIN),	\
				MIN_DOMAIN									\
			) - MIN_DOMAIN									\
		) *													\
		fmaf(dimensions[x], DOMAIN_DELTA, -DOMAIN_DELTA)	\
	)

	int temp = __COORD(0);

#pragma unroll
	for (int i = 1; i < DIMS; ++i) {
		temp = (int)dimensions[i] * temp + __COORD(i);
	}

	result[ind] = image[temp];
}
